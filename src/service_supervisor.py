from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Any
from urllib import error, parse, request


class LifecycleMode(str, Enum):
    EXTERNAL = 'external'
    MANAGED_LOCAL_SERVER = 'managed_local_server'
    MANAGED_LLAMA_SERVER = 'managed_llama_server'


class ServiceState(str, Enum):
    UNCONFIGURED = 'unconfigured'
    EXTERNAL = 'external'
    MANAGED_STOPPED = 'managed_stopped'
    MANAGED_STARTING = 'managed_starting'
    MANAGED_READY = 'managed_ready'
    MANAGED_UNHEALTHY = 'managed_unhealthy'
    FOREIGN_PORT_OWNER = 'foreign_port_owner'
    UNKNOWN = 'unknown'


class BackendHealth(str, Enum):
    NOT_CHECKED = 'not_checked'
    HEALTHY = 'healthy'
    UNREACHABLE = 'unreachable'
    TIMEOUT = 'timeout'
    AUTH_FAILED = 'auth_failed'
    MODEL_MISSING = 'model_missing'
    MODELS_ENDPOINT_UNSUPPORTED = 'models_endpoint_unsupported'
    MALFORMED_RESPONSE = 'malformed_response'
    PROVIDER_ERROR = 'provider_error'
    UNKNOWN = 'unknown'


class ErrorKind(str, Enum):
    UNREACHABLE = 'unreachable'
    TIMEOUT = 'timeout'
    AUTH_FAILED = 'auth_failed'
    MODEL_MISSING = 'model_missing'
    MODELS_ENDPOINT_UNSUPPORTED = 'models_endpoint_unsupported'
    MALFORMED_RESPONSE = 'malformed_response'
    PROVIDER_ERROR = 'provider_error'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class ServiceConfig:
    role: str
    base_url: str | None = None
    model_id: str | None = None
    api_key: str | None = None
    lifecycle_mode: LifecycleMode = LifecycleMode.EXTERNAL
    host: str = '127.0.0.1'
    port: int | None = None
    runtime_dir: Path | None = None
    pid_file: Path | None = None
    log_file: Path | None = None
    owner_marker_file: Path | None = None
    executable: str | None = None
    model_path: Path | None = None
    context_size: int | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    gpu: str | None = None
    extra_args: tuple[str, ...] = ()
    lifecycle_enabled: bool = True


@dataclass(frozen=True)
class ServiceHealth:
    role: str
    state: ServiceState
    backend_health: BackendHealth = BackendHealth.NOT_CHECKED
    error_kind: ErrorKind | None = None
    message: str = ''
    base_url: str | None = None
    model_id: str | None = None
    model_present: bool | None = None
    pid: int | None = None
    port: int | None = None
    lifecycle_controllable: bool = False
    raw_models: tuple[str, ...] = ()


UrlOpen = Callable[..., Any]
PortPids = Callable[[int], tuple[int, ...]]
ProcessAlive = Callable[[int], bool]
CommandLine = Callable[[int], str | None]


class ServiceSupervisor:
    def __init__(
        self,
        config: ServiceConfig,
        *,
        urlopen: UrlOpen | None = None,
        port_pids: PortPids | None = None,
        process_alive: ProcessAlive | None = None,
        command_line: CommandLine | None = None,
        timeout_seconds: float = 2.0,
    ) -> None:
        self.config = config
        self._urlopen = urlopen or request.urlopen
        self._port_pids = port_pids or _port_pids
        self._process_alive = process_alive or _process_alive
        self._command_line = command_line or _command_line
        self._timeout_seconds = timeout_seconds

    def check(self) -> ServiceHealth:
        if not self.config.base_url and self.config.lifecycle_mode is LifecycleMode.EXTERNAL:
            return ServiceHealth(
                role=self.config.role,
                state=ServiceState.UNCONFIGURED,
                base_url=self.config.base_url,
                model_id=self.config.model_id,
                port=self.config.port,
            )

        if self.config.lifecycle_mode is LifecycleMode.EXTERNAL:
            backend = self._check_backend()
            return ServiceHealth(
                role=self.config.role,
                state=ServiceState.EXTERNAL,
                backend_health=backend.backend_health,
                error_kind=backend.error_kind,
                message=backend.message,
                base_url=self.config.base_url,
                model_id=self.config.model_id,
                model_present=backend.model_present,
                port=self.config.port,
                lifecycle_controllable=False,
                raw_models=backend.raw_models,
            )

        owned_pid = self._owned_pid()
        candidate_pid = self._candidate_pid()
        port_owners = self._port_owners()
        if owned_pid is None:
            if candidate_pid is not None and (not port_owners or port_owners == (candidate_pid,)):
                return ServiceHealth(
                    role=self.config.role,
                    state=ServiceState.UNKNOWN,
                    backend_health=BackendHealth.NOT_CHECKED,
                    message='PID file points to a live matching process, but supervisor ownership marker is missing or invalid.',
                    base_url=self.config.base_url,
                    model_id=self.config.model_id,
                    pid=candidate_pid,
                    port=self.config.port,
                    lifecycle_controllable=False,
                )
            if port_owners:
                return ServiceHealth(
                    role=self.config.role,
                    state=ServiceState.FOREIGN_PORT_OWNER,
                    backend_health=BackendHealth.NOT_CHECKED,
                    message='Configured port is owned by a process not proven to belong to Dragon Clawed TUI.',
                    base_url=self.config.base_url,
                    model_id=self.config.model_id,
                    pid=next((pid for pid in port_owners if pid != candidate_pid), port_owners[0]),
                    port=self.config.port,
                    lifecycle_controllable=False,
                )
            return ServiceHealth(
                role=self.config.role,
                state=ServiceState.MANAGED_STOPPED,
                backend_health=BackendHealth.NOT_CHECKED,
                base_url=self.config.base_url,
                model_id=self.config.model_id,
                port=self.config.port,
                lifecycle_controllable=False,
            )

        if self.config.port is None:
            return ServiceHealth(
                role=self.config.role,
                state=ServiceState.UNKNOWN,
                backend_health=BackendHealth.NOT_CHECKED,
                message='Owned PID exists, but no configured port exists to verify listener ownership.',
                base_url=self.config.base_url,
                model_id=self.config.model_id,
                pid=owned_pid,
                port=self.config.port,
                lifecycle_controllable=False,
            )

        if port_owners and any(pid != owned_pid for pid in port_owners):
            return ServiceHealth(
                role=self.config.role,
                state=ServiceState.FOREIGN_PORT_OWNER,
                backend_health=BackendHealth.NOT_CHECKED,
                message='PID file points to an owned process, but the configured port is owned by another process.',
                base_url=self.config.base_url,
                model_id=self.config.model_id,
                pid=next(pid for pid in port_owners if pid != owned_pid),
                port=self.config.port,
                lifecycle_controllable=False,
            )
        if not port_owners and self.config.port is not None:
            return ServiceHealth(
                role=self.config.role,
                state=ServiceState.UNKNOWN,
                backend_health=BackendHealth.NOT_CHECKED,
                message='Owned PID exists, but listener ownership could not be verified.',
                base_url=self.config.base_url,
                model_id=self.config.model_id,
                pid=owned_pid,
                port=self.config.port,
                lifecycle_controllable=False,
            )

        backend = self._check_backend()
        state = (
            ServiceState.MANAGED_READY
            if backend.backend_health in (BackendHealth.HEALTHY, BackendHealth.MODELS_ENDPOINT_UNSUPPORTED)
            else ServiceState.MANAGED_UNHEALTHY
        )
        return ServiceHealth(
            role=self.config.role,
            state=state,
            backend_health=backend.backend_health,
            error_kind=backend.error_kind,
            message=backend.message,
            base_url=self.config.base_url,
            model_id=self.config.model_id,
            model_present=backend.model_present,
            pid=owned_pid,
            port=self.config.port,
            lifecycle_controllable=self.config.lifecycle_enabled,
            raw_models=backend.raw_models,
        )

    def _check_backend(self) -> ServiceHealth:
        if not self.config.base_url:
            return ServiceHealth(
                role=self.config.role,
                state=ServiceState.UNCONFIGURED,
                backend_health=BackendHealth.NOT_CHECKED,
            )
        url = _join_url(self.config.base_url, '/models')
        req = request.Request(url, headers=_auth_headers(self.config))
        try:
            with self._urlopen(req, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read().decode('utf-8'))
        except error.HTTPError as exc:
            if exc.code in (401, 403):
                return _backend_result(self.config, BackendHealth.AUTH_FAILED, ErrorKind.AUTH_FAILED, f'HTTP {exc.code}')
            if exc.code == 404:
                return _backend_result(
                    self.config,
                    BackendHealth.MODELS_ENDPOINT_UNSUPPORTED,
                    ErrorKind.MODELS_ENDPOINT_UNSUPPORTED,
                    'models endpoint unsupported',
                )
            return _backend_result(self.config, BackendHealth.PROVIDER_ERROR, ErrorKind.PROVIDER_ERROR, f'HTTP {exc.code}')
        except TimeoutError:
            return _backend_result(self.config, BackendHealth.TIMEOUT, ErrorKind.TIMEOUT, 'backend health probe timed out')
        except error.URLError as exc:
            return _backend_result(self.config, BackendHealth.UNREACHABLE, ErrorKind.UNREACHABLE, str(exc.reason))
        except OSError as exc:
            return _backend_result(self.config, BackendHealth.UNREACHABLE, ErrorKind.UNREACHABLE, str(exc))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return _backend_result(self.config, BackendHealth.MALFORMED_RESPONSE, ErrorKind.MALFORMED_RESPONSE, str(exc))

        models = _extract_model_ids(payload)
        if models is None:
            return _backend_result(self.config, BackendHealth.MALFORMED_RESPONSE, ErrorKind.MALFORMED_RESPONSE, 'models response did not contain model ids')
        if self.config.model_id and self.config.model_id not in models:
            result = _backend_result(self.config, BackendHealth.MODEL_MISSING, ErrorKind.MODEL_MISSING, 'target model missing')
            return ServiceHealth(**{**result.__dict__, 'model_present': False, 'raw_models': tuple(models)})
        result = _backend_result(self.config, BackendHealth.HEALTHY, None, 'backend healthy')
        return ServiceHealth(**{**result.__dict__, 'model_present': bool(self.config.model_id), 'raw_models': tuple(models)})

    def _candidate_pid(self) -> int | None:
        pid = _read_pid_file(self.config.pid_file)
        if pid is None:
            return None
        if not self._process_alive(pid):
            return None
        command = self._command_line(pid) or ''
        if not self.config.executable or not _command_matches_executable(command, self.config.executable):
            return None
        return pid

    def _owned_pid(self) -> int | None:
        pid = self._candidate_pid()
        if pid is None:
            return None
        if not _owner_marker_matches(self.config.owner_marker_file, self.config.role, pid):
            return None
        return pid

    def _port_owners(self) -> tuple[int, ...]:
        if self.config.port is None:
            return ()
        return self._port_pids(self.config.port)


def supervisor_configs_from_env(env: Mapping[str, str] | None = None) -> dict[str, ServiceConfig]:
    source = dict(os.environ if env is None else env)
    runtime_dir = Path(source.get('CLAWED_SERVICE_RUNTIME_DIR') or source.get('XDG_RUNTIME_DIR') or '/tmp') / 'dragon-clawed-tui'
    main_port = _optional_int(source.get('CLAWED_MAIN_PORT'))
    helper_port = _optional_int(source.get('CLAWED_HELPER_PORT'))
    return {
        'main': _config_from_env(
            source,
            role='main',
            base_url=source.get('OPENAI_BASE_URL'),
            model_id=source.get('OPENAI_MODEL'),
            api_key=source.get('OPENAI_API_KEY'),
            port=main_port,
            runtime_dir=runtime_dir,
        ),
        'helper': _config_from_env(
            source,
            role='helper',
            base_url=source.get('CLAWED_HELPER_BASE_URL') or source.get('CLAWED_DELEGATE_BASE_URL'),
            model_id=source.get('CLAWED_HELPER_MODEL_ID') or source.get('CLAWED_DELEGATE_MODEL'),
            api_key=source.get('CLAWED_HELPER_API_KEY') or source.get('CLAWED_DELEGATE_API_KEY'),
            port=helper_port,
            runtime_dir=runtime_dir,
        ),
    }


def _config_from_env(
    env: Mapping[str, str],
    *,
    role: str,
    base_url: str | None,
    model_id: str | None,
    api_key: str | None,
    port: int | None,
    runtime_dir: Path,
) -> ServiceConfig:
    prefix = 'CLAWED_MAIN' if role == 'main' else 'CLAWED_HELPER'
    lifecycle_raw = env.get(f'{prefix}_LIFECYCLE')
    model_path_raw = env.get(f'{prefix}_MODEL')
    lifecycle_mode = _lifecycle_mode(lifecycle_raw)
    host = env.get(f'{prefix}_HOST', '127.0.0.1')
    if base_url is None and port is not None:
        base_url = f'http://{host}:{port}/v1'
    return ServiceConfig(
        role=role,
        base_url=base_url,
        model_id=model_id,
        api_key=api_key,
        lifecycle_mode=lifecycle_mode,
        host=host,
        port=port,
        runtime_dir=runtime_dir,
        pid_file=runtime_dir / f'{role}.pid',
        log_file=runtime_dir / f'{role}.log',
        owner_marker_file=runtime_dir / f'{role}.owner',
        executable=env.get('CLAWED_SERVER_BIN') or env.get('CLAWED_LLAMA_SERVER_BIN'),
        model_path=Path(model_path_raw) if model_path_raw else None,
        context_size=_optional_int(env.get(f'{prefix}_CONTEXT_SIZE') or env.get('CLAWED_CONTEXT_SIZE')),
        cache_type_k=env.get('CLAWED_CACHE_TYPE_K'),
        cache_type_v=env.get('CLAWED_CACHE_TYPE_V'),
        gpu=env.get(f'{prefix}_GPU'),
        extra_args=tuple((env.get('CLAWED_SERVER_EXTRA_ARGS') or '').split()),
        lifecycle_enabled=_truthy(env.get(f'{prefix}_LIFECYCLE_ENABLED'), default=True),
    )


def _lifecycle_mode(raw: str | None) -> LifecycleMode:
    if raw == LifecycleMode.MANAGED_LOCAL_SERVER.value:
        return LifecycleMode.MANAGED_LOCAL_SERVER
    if raw == LifecycleMode.MANAGED_LLAMA_SERVER.value:
        return LifecycleMode.MANAGED_LOCAL_SERVER
    if raw == LifecycleMode.EXTERNAL.value:
        return LifecycleMode.EXTERNAL
    return LifecycleMode.EXTERNAL


def _auth_headers(config: ServiceConfig) -> dict[str, str]:
    if config.api_key:
        return {'Authorization': f'Bearer {config.api_key}'}
    if config.base_url and _is_loopback_url(config.base_url):
        return {'Authorization': 'Bearer local-token'}
    return {}


def _is_loopback_url(url: str) -> bool:
    host = parse.urlparse(url).hostname
    return host in {'127.0.0.1', 'localhost', '::1'}


def _owner_marker_matches(path: Path | None, role: str, pid: int) -> bool:
    if path is None or not path.exists():
        return False
    try:
        parts = path.read_text(encoding='utf-8').split()
    except OSError:
        return False
    return len(parts) >= 3 and parts[0] == 'dragon-clawed-tui' and parts[1] == role and parts[2] == str(pid)


def _backend_result(config: ServiceConfig, health: BackendHealth, kind: ErrorKind | None, message: str) -> ServiceHealth:
    return ServiceHealth(
        role=config.role,
        state=ServiceState.UNKNOWN,
        backend_health=health,
        error_kind=kind,
        message=message,
        base_url=config.base_url,
        model_id=config.model_id,
        port=config.port,
    )


def _extract_model_ids(payload: Any) -> list[str] | None:
    if not isinstance(payload, dict):
        return None
    raw_models = payload.get('data') or payload.get('models')
    if not isinstance(raw_models, list):
        return None
    ids: list[str] = []
    for item in raw_models:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict):
            model_id = item.get('id') or item.get('name') or item.get('model')
            if isinstance(model_id, str):
                ids.append(model_id)
    return ids


def _join_url(base_url: str, suffix: str) -> str:
    return f'{base_url.rstrip("/")}/{suffix.lstrip("/")}'


def _read_pid_file(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    try:
        return int(path.read_text(encoding='utf-8').strip())
    except (OSError, ValueError):
        return None


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _command_line(pid: int) -> str | None:
    result = subprocess.run(
        ['ps', '-p', str(pid), '-o', 'args='],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _command_matches_executable(command: str, executable: str) -> bool:
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    if not argv:
        return False
    expected = Path(executable).name
    actual = Path(argv[0]).name
    return actual == expected


def _port_pids(port: int) -> tuple[int, ...]:
    for command in (
        ['lsof', '-tiTCP:%d' % port, '-sTCP:LISTEN'],
        ['ss', '-ltnp', 'sport = :%d' % port],
    ):
        try:
            result = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except FileNotFoundError:
            continue
        if result.returncode != 0 or not result.stdout.strip():
            continue
        if command[0] == 'lsof':
            pids = _parse_lsof_pids(result.stdout)
        else:
            pids = _parse_ss_pids(result.stdout)
        if pids:
            return pids
    return ()


def _parse_lsof_pids(output: str) -> tuple[int, ...]:
    pids: list[int] = []
    for line in output.splitlines():
        try:
            pids.append(int(line.strip()))
        except ValueError:
            continue
    return tuple(dict.fromkeys(pids))


def _parse_ss_pids(output: str) -> tuple[int, ...]:
    import re

    return tuple(dict.fromkeys(int(match) for match in re.findall(r'pid=(\d+)', output)))


def _optional_int(raw: str | None) -> int | None:
    if raw is None or raw == '':
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _truthy(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    return raw.lower() in {'1', 'true', 'yes', 'on'}
