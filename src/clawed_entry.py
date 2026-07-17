from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .main import _detect_ollama_model, _load_dotenv_defaults


@dataclass
class ReplState:
    cwd: Path
    session_id: str = ''
    allow_write: bool = False
    allow_shell: bool = False
    unsafe: bool = False
    max_turns: int = 3
    auto_snip_threshold: int = 4800
    auto_compact_threshold: int = 5200
    compact_preserve_messages: int = 1
    max_tool_calls: int = 4
    max_delegated_tasks: int = 0

    @property
    def session_file(self) -> Path:
        return self.cwd / '.port_sessions' / 'launcher_last_session_id'


def _print_banner(state: ReplState) -> None:
    print('Claw Code Agent (Python Claude Code-style CLI)')
    print(f'CWD: {state.cwd}')
    print(f'Python: {sys.executable}')
    print()
    print('Commands:')
    print('  :help        show help')
    print('  :new         start a fresh session')
    print('  :resume-last resume the last saved launcher session')
    print('  :write       toggle --allow-write')
    print('  :shell       toggle --allow-shell')
    print('  :unsafe      toggle --unsafe')
    print('  :limits      show Groq token guardrails')
    print('  !<cmd>       run a shell command locally')
    print('  :quit        exit')
    print()
    print('Type a prompt to run the agent. New launcher windows start fresh.')


def _print_status(state: ReplState) -> None:
    flags: list[str] = []
    if state.allow_write:
        flags.append('--allow-write')
    if state.allow_shell:
        flags.append('--allow-shell')
    if state.unsafe:
        flags.append('--unsafe')
    print()
    print(f'[session] {state.session_id or "<new>"}')
    print(f'[flags]   {" ".join(flags) if flags else "<none>"}')
    print(
        f'[limits]  max_turns={state.max_turns} '
        f'compact_at={state.auto_compact_threshold} '
        f'snip_at={state.auto_snip_threshold} '
        f'preserve={state.compact_preserve_messages} '
        f'max_tool_calls={state.max_tool_calls}'
    )
    print()


def _load_session_id(state: ReplState) -> None:
    try:
        if state.session_file.exists():
            state.session_id = state.session_file.read_text(encoding='utf-8').splitlines()[0].strip()
    except OSError:
        return


def _persist_session_id(state: ReplState, session_id: str) -> None:
    if not session_id:
        return
    try:
        state.session_file.parent.mkdir(parents=True, exist_ok=True)
        state.session_file.write_text(f'{session_id}\n', encoding='utf-8')
        state.session_id = session_id
    except OSError:
        state.session_id = session_id


def _run_agent_once(state: ReplState, prompt: str) -> int:
    args: list[str] = [sys.executable, '-m', 'src.main']
    if state.session_id:
        args += ['agent-resume', state.session_id, prompt]
    else:
        args += ['agent', prompt]

    args += [
        '--cwd', str(state.cwd),
        '--max-turns', str(state.max_turns),
        '--auto-snip-threshold', str(state.auto_snip_threshold),
        '--auto-compact-threshold', str(state.auto_compact_threshold),
        '--compact-preserve-messages', str(state.compact_preserve_messages),
        '--max-tool-calls', str(state.max_tool_calls),
        '--max-delegated-tasks', str(state.max_delegated_tasks),
    ]
    base_url = os.environ.get('OPENAI_BASE_URL')
    api_key = os.environ.get('OPENAI_API_KEY')
    model = os.environ.get('OPENAI_MODEL') or _detect_ollama_model()
    if base_url:
        args += ['--base-url', base_url]
    if api_key:
        args += ['--api-key', api_key]
    if model:
        args += ['--model', model]
    if state.allow_write:
        args.append('--allow-write')
    if state.allow_shell:
        args.append('--allow-shell')
    if state.unsafe:
        args.append('--unsafe')

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    assert proc.stdout is not None

    last_session_id = ''
    for line in proc.stdout:
        sys.stdout.write(line)
        if line.startswith('session_id='):
            last_session_id = line.split('=', 1)[1].strip()

    exit_code = proc.wait()
    if last_session_id:
        _persist_session_id(state, last_session_id)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='clawed', add_help=True)
    parser.add_argument('--cwd', default='.', help='workspace directory to run in')
    args = parser.parse_args(argv)

    state = ReplState(cwd=Path(args.cwd).resolve())
    os.chdir(state.cwd)
    _load_dotenv_defaults()
    _print_banner(state)
    _print_status(state)

    while True:
        try:
            line = input('claw> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line.startswith('!'):
            command = line[1:].strip()
            if not command:
                continue
            subprocess.run(command, shell=True, cwd=str(state.cwd))
            continue
        if line in (':quit', ':q', 'quit', 'exit'):
            break
        if line in (':help', 'help'):
            _print_banner(state)
            _print_status(state)
            continue
        if line == ':limits':
            _print_status(state)
            continue
        if line == ':new':
            state.session_id = ''
            _print_status(state)
            continue
        if line == ':resume-last':
            _load_session_id(state)
            _print_status(state)
            continue
        if line == ':write':
            state.allow_write = not state.allow_write
            _print_status(state)
            continue
        if line == ':shell':
            state.allow_shell = not state.allow_shell
            _print_status(state)
            continue
        if line == ':unsafe':
            state.unsafe = not state.unsafe
            _print_status(state)
            continue

        exit_code = _run_agent_once(state, line)
        if exit_code != 0:
            print()
            print(f'[exit] {exit_code}')
            print()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
