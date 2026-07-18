from __future__ import annotations

import json
from pathlib import Path
from urllib import error, request

from src.service_supervisor import (
    BackendHealth,
    ErrorKind,
    LifecycleMode,
    ServiceConfig,
    ServiceState,
    ServiceSupervisor,
    supervisor_configs_from_env,
)
from src.service_supervisor import _parse_lsof_pids, _parse_ss_pids


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def read(self) -> bytes:
        return json.dumps(self.payload).encode('utf-8')

    def __enter__(self) -> 'FakeHTTPResponse':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_external_backend_reports_healthy_when_target_model_is_listed() -> None:
    def fake_urlopen(request_obj, timeout=None):  # noqa: ANN001
        assert request_obj.full_url == 'http://127.0.0.1:8000/v1/models'
        assert request_obj.headers['Authorization'] == 'Bearer local-token'
        return FakeHTTPResponse({'data': [{'id': 'local-model'}]})

    supervisor = ServiceSupervisor(
        ServiceConfig(
            role='main',
            base_url='http://127.0.0.1:8000/v1',
            model_id='local-model',
            lifecycle_mode=LifecycleMode.EXTERNAL,
        ),
        urlopen=fake_urlopen,
    )

    health = supervisor.check()

    assert health.state is ServiceState.EXTERNAL
    assert health.backend_health is BackendHealth.HEALTHY
    assert health.error_kind is None
    assert health.model_present is True
    assert health.lifecycle_controllable is False


def test_external_backend_auth_failure_is_not_reported_as_service_down() -> None:
    def fake_urlopen(request_obj, timeout=None):  # noqa: ANN001
        raise error.HTTPError(request_obj.full_url, 401, 'unauthorized', {}, None)

    supervisor = ServiceSupervisor(
        ServiceConfig(
            role='main',
            base_url='https://provider.example/v1',
            model_id='remote-model',
            lifecycle_mode=LifecycleMode.EXTERNAL,
        ),
        urlopen=fake_urlopen,
    )

    health = supervisor.check()

    assert health.state is ServiceState.EXTERNAL
    assert health.backend_health is BackendHealth.AUTH_FAILED
    assert health.error_kind is ErrorKind.AUTH_FAILED
    assert health.lifecycle_controllable is False


def test_models_endpoint_unsupported_is_unknown_not_down() -> None:
    def fake_urlopen(request_obj, timeout=None):  # noqa: ANN001
        raise error.HTTPError(request_obj.full_url, 404, 'not found', {}, None)

    supervisor = ServiceSupervisor(
        ServiceConfig(
            role='main',
            base_url='https://provider.example/v1',
            model_id='remote-model',
            lifecycle_mode=LifecycleMode.EXTERNAL,
        ),
        urlopen=fake_urlopen,
    )

    health = supervisor.check()

    assert health.state is ServiceState.EXTERNAL
    assert health.backend_health is BackendHealth.MODELS_ENDPOINT_UNSUPPORTED
    assert health.error_kind is ErrorKind.MODELS_ENDPOINT_UNSUPPORTED


def test_managed_service_without_pid_or_port_owner_is_stopped(tmp_path: Path) -> None:
    supervisor = ServiceSupervisor(
        ServiceConfig(
            role='main',
            base_url='http://127.0.0.1:8080/v1',
            model_id='local-model',
            lifecycle_mode=LifecycleMode.MANAGED_LOCAL_SERVER,
            port=8080,
            pid_file=tmp_path / 'main.pid',
            executable='llama-server',
        ),
        port_pids=lambda port: (),
    )

    health = supervisor.check()

    assert health.state is ServiceState.MANAGED_STOPPED
    assert health.lifecycle_controllable is False


def test_managed_service_refuses_foreign_port_owner(tmp_path: Path) -> None:
    supervisor = ServiceSupervisor(
        ServiceConfig(
            role='main',
            base_url='http://127.0.0.1:8080/v1',
            model_id='local-model',
            lifecycle_mode=LifecycleMode.MANAGED_LOCAL_SERVER,
            port=8080,
            pid_file=tmp_path / 'main.pid',
            executable='llama-server',
        ),
        port_pids=lambda port: (999,),
        process_alive=lambda pid: True,
        command_line=lambda pid: 'python -m http.server 8080',
    )

    health = supervisor.check()

    assert health.state is ServiceState.FOREIGN_PORT_OWNER
    assert health.lifecycle_controllable is False
    assert health.pid == 999


def test_managed_service_with_owned_pid_and_matching_model_is_ready(tmp_path: Path) -> None:
    pid_file = tmp_path / 'main.pid'
    marker_file = tmp_path / 'main.owner'
    pid_file.write_text('123', encoding='utf-8')
    marker_file.write_text('dragon-clawed-tui main 123', encoding='utf-8')

    def fake_urlopen(request_obj, timeout=None):  # noqa: ANN001
        return FakeHTTPResponse({'data': [{'id': 'local-model'}]})

    supervisor = ServiceSupervisor(
        ServiceConfig(
            role='main',
            base_url='http://127.0.0.1:8080/v1',
            model_id='local-model',
            lifecycle_mode=LifecycleMode.MANAGED_LOCAL_SERVER,
            port=8080,
            pid_file=pid_file,
            executable='llama-server',
            owner_marker_file=marker_file,
        ),
        urlopen=fake_urlopen,
        port_pids=lambda port: (123,),
        process_alive=lambda pid: True,
        command_line=lambda pid: '/usr/bin/llama-server --port 8080',
    )

    health = supervisor.check()

    assert health.state is ServiceState.MANAGED_READY
    assert health.backend_health is BackendHealth.HEALTHY
    assert health.lifecycle_controllable is True
    assert health.pid == 123


def test_supervisor_configs_from_env_preserves_generic_backend_settings() -> None:
    configs = supervisor_configs_from_env(
        {
            'OPENAI_BASE_URL': 'http://127.0.0.1:8000/v1',
            'OPENAI_MODEL': 'main-model',
            'CLAWED_HELPER_BASE_URL': 'http://127.0.0.1:8001/v1',
            'CLAWED_HELPER_MODEL_ID': 'helper-model',
            'CLAWED_MAIN_LIFECYCLE': 'external',
            'CLAWED_HELPER_LIFECYCLE': 'managed_local_server',
            'CLAWED_HELPER_MODEL': '/models/helper.gguf',
            'CLAWED_HELPER_PORT': '8001',
            'CLAWED_SERVICE_RUNTIME_DIR': '/tmp/dragon-clawed-test',
        }
    )

    assert configs['main'].base_url == 'http://127.0.0.1:8000/v1'
    assert configs['main'].model_id == 'main-model'
    assert configs['main'].lifecycle_mode is LifecycleMode.EXTERNAL
    assert configs['helper'].base_url == 'http://127.0.0.1:8001/v1'
    assert configs['helper'].model_id == 'helper-model'
    assert configs['helper'].model_path == Path('/models/helper.gguf')
    assert configs['helper'].port == 8001
    assert configs['helper'].lifecycle_mode is LifecycleMode.MANAGED_LOCAL_SERVER


def test_managed_service_refuses_when_any_foreign_process_shares_port(tmp_path: Path) -> None:
    pid_file = tmp_path / 'main.pid'
    pid_file.write_text('123', encoding='utf-8')

    supervisor = ServiceSupervisor(
        ServiceConfig(
            role='main',
            base_url='http://127.0.0.1:8080/v1',
            model_id='local-model',
            lifecycle_mode=LifecycleMode.MANAGED_LOCAL_SERVER,
            port=8080,
            pid_file=pid_file,
            executable='llama-server',
        ),
        port_pids=lambda port: (123, 999),
        process_alive=lambda pid: True,
        command_line=lambda pid: '/usr/bin/llama-server --port 8080',
    )

    health = supervisor.check()

    assert health.state is ServiceState.FOREIGN_PORT_OWNER
    assert health.lifecycle_controllable is False
    assert health.pid == 999


def test_managed_service_requires_exact_executable_identity(tmp_path: Path) -> None:
    pid_file = tmp_path / 'main.pid'
    pid_file.write_text('123', encoding='utf-8')

    supervisor = ServiceSupervisor(
        ServiceConfig(
            role='main',
            base_url='http://127.0.0.1:8080/v1',
            model_id='local-model',
            lifecycle_mode=LifecycleMode.MANAGED_LOCAL_SERVER,
            port=8080,
            pid_file=pid_file,
            executable='llama-server',
        ),
        port_pids=lambda port: (123,),
        process_alive=lambda pid: True,
        command_line=lambda pid: '/usr/bin/not-llama-server --port 8080',
    )

    health = supervisor.check()

    assert health.state is ServiceState.FOREIGN_PORT_OWNER
    assert health.lifecycle_controllable is False


def test_port_pid_parsers_return_all_unique_listener_pids() -> None:
    assert _parse_lsof_pids('123\n999\n123\nnot-a-pid\n') == (123, 999)
    assert _parse_ss_pids('users:(("server",pid=123,fd=7),("other",pid=999,fd=8))') == (123, 999)


def test_external_backend_uses_configured_api_key_for_health_probe() -> None:
    seen_headers: list[str] = []

    def fake_urlopen(request_obj: request.Request, timeout=None):  # noqa: ANN001
        seen_headers.append(request_obj.headers['Authorization'])
        return FakeHTTPResponse({'data': [{'id': 'remote-model'}]})

    supervisor = ServiceSupervisor(
        ServiceConfig(
            role='main',
            base_url='https://provider.example/v1',
            api_key='provider-key',
            model_id='remote-model',
            lifecycle_mode=LifecycleMode.EXTERNAL,
        ),
        urlopen=fake_urlopen,
    )

    health = supervisor.check()

    assert health.backend_health is BackendHealth.HEALTHY
    assert seen_headers == ['Bearer provider-key']


def test_model_path_without_explicit_managed_lifecycle_remains_external() -> None:
    configs = supervisor_configs_from_env(
        {
            'OPENAI_BASE_URL': 'http://127.0.0.1:8000/v1',
            'OPENAI_MODEL': 'main-model',
            'CLAWED_MAIN_MODEL': '/models/main.gguf',
        }
    )

    assert configs['main'].lifecycle_mode is LifecycleMode.EXTERNAL
    assert configs['main'].model_path == Path('/models/main.gguf')


def test_managed_service_with_same_binary_but_no_owner_marker_is_not_controllable(tmp_path: Path) -> None:
    pid_file = tmp_path / 'main.pid'
    pid_file.write_text('123', encoding='utf-8')

    supervisor = ServiceSupervisor(
        ServiceConfig(
            role='main',
            base_url='http://127.0.0.1:8080/v1',
            model_id='local-model',
            lifecycle_mode=LifecycleMode.MANAGED_LOCAL_SERVER,
            port=8080,
            pid_file=pid_file,
            executable='llama-server',
        ),
        port_pids=lambda port: (123,),
        process_alive=lambda pid: True,
        command_line=lambda pid: '/usr/bin/llama-server --port 8080',
    )

    health = supervisor.check()

    assert health.state is ServiceState.UNKNOWN
    assert health.lifecycle_controllable is False


def test_managed_service_with_no_port_is_not_controllable_even_with_owner_marker(tmp_path: Path) -> None:
    pid_file = tmp_path / 'main.pid'
    marker_file = tmp_path / 'main.owner'
    pid_file.write_text('123', encoding='utf-8')
    marker_file.write_text('dragon-clawed-tui main 123', encoding='utf-8')

    supervisor = ServiceSupervisor(
        ServiceConfig(
            role='main',
            base_url='http://127.0.0.1:8080/v1',
            model_id='local-model',
            lifecycle_mode=LifecycleMode.MANAGED_LOCAL_SERVER,
            pid_file=pid_file,
            owner_marker_file=marker_file,
            executable='llama-server',
        ),
        process_alive=lambda pid: True,
        command_line=lambda pid: '/usr/bin/llama-server --port 8080',
    )

    health = supervisor.check()

    assert health.state is ServiceState.UNKNOWN
    assert health.lifecycle_controllable is False


def test_helper_config_does_not_reuse_main_api_key_for_different_endpoint() -> None:
    configs = supervisor_configs_from_env(
        {
            'OPENAI_BASE_URL': 'https://main-provider.example/v1',
            'OPENAI_API_KEY': 'main-key',
            'OPENAI_MODEL': 'main-model',
            'CLAWED_HELPER_BASE_URL': 'https://helper-provider.example/v1',
            'CLAWED_HELPER_MODEL_ID': 'helper-model',
        }
    )

    assert configs['main'].api_key == 'main-key'
    assert configs['helper'].api_key is None


def test_env_managed_service_requires_configured_server_executable() -> None:
    configs = supervisor_configs_from_env(
        {
            'CLAWED_HELPER_LIFECYCLE': 'managed_local_server',
            'CLAWED_HELPER_MODEL': '/models/helper.gguf',
            'CLAWED_HELPER_PORT': '8001',
        }
    )

    assert configs['helper'].lifecycle_mode is LifecycleMode.MANAGED_LOCAL_SERVER
    assert configs['helper'].executable is None


def test_external_non_loopback_without_api_key_sends_no_dummy_auth() -> None:
    seen_headers: list[str | None] = []

    def fake_urlopen(request_obj: request.Request, timeout=None):  # noqa: ANN001
        seen_headers.append(request_obj.headers.get('Authorization'))
        return FakeHTTPResponse({'data': [{'id': 'remote-model'}]})

    supervisor = ServiceSupervisor(
        ServiceConfig(
            role='main',
            base_url='https://provider.example/v1',
            model_id='remote-model',
            lifecycle_mode=LifecycleMode.EXTERNAL,
        ),
        urlopen=fake_urlopen,
    )

    health = supervisor.check()

    assert health.backend_health is BackendHealth.HEALTHY
    assert seen_headers == [None]


def test_legacy_managed_llama_server_lifecycle_alias_maps_to_managed_local() -> None:
    configs = supervisor_configs_from_env(
        {
            'CLAWED_HELPER_LIFECYCLE': 'managed_llama_server',
            'CLAWED_HELPER_PORT': '8001',
            'CLAWED_SERVER_BIN': '/usr/local/bin/custom-server',
        }
    )

    assert configs['helper'].lifecycle_mode is LifecycleMode.MANAGED_LOCAL_SERVER
    assert configs['helper'].executable == '/usr/local/bin/custom-server'
