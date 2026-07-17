from __future__ import annotations

from dataclasses import dataclass, replace
import curses
import subprocess
import textwrap
from typing import Callable, Protocol

from .agent_runtime import LocalCodingAgent
from .agent_types import (
    AgentPermissions,
    AgentRunResult,
    AgentRuntimeConfig,
    ModelConfig,
)
from .session_store import AgentSessionSummary, StoredAgentSession, list_agent_sessions, load_agent_session
from .voice_input import VoiceInputService, VoiceRecording


class AgentFactory(Protocol):
    def __call__(
        self,
        model_config: ModelConfig,
        runtime_config: AgentRuntimeConfig,
    ) -> LocalCodingAgent:
        ...


@dataclass
class TuiController:
    model_config: ModelConfig
    runtime_config: AgentRuntimeConfig
    custom_system_prompt: str | None = None
    append_system_prompt: str | None = None
    override_system_prompt: str | None = None
    agent_factory: AgentFactory | None = None
    voice_input_service: VoiceInputService | None = None
    session_id: str | None = None
    session_path: str | None = None
    last_result: AgentRunResult | None = None
    status: str = 'ready'
    last_session_picker: tuple[AgentSessionSummary, ...] | None = None
    last_session_filter: str = ''

    def run_prompt(
        self,
        prompt: str,
        *,
        event_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> AgentRunResult:
        self.status = 'running'
        agent = self._build_agent()
        try:
            if self.session_id:
                stored_session = self._load_session(self.session_id)
                result = agent.resume(prompt, stored_session, event_callback=event_callback)
            else:
                result = agent.run(prompt, event_callback=event_callback)
        finally:
            self.status = 'ready'
        self.last_result = result
        if result.session_id:
            self.session_id = result.session_id
        if result.session_path:
            self.session_path = result.session_path
        return result

    def start_new_session(self) -> None:
        self.session_id = None
        self.session_path = None
        self.last_result = None
        self.status = 'ready'

    def resume_session(self, session_id: str) -> None:
        self.session_id = session_id.strip() or None
        self.session_path = None
        self.last_result = None
        self.status = 'ready'

    def recent_sessions(self, *, limit: int = 10) -> tuple[AgentSessionSummary, ...]:
        return list_agent_sessions(self.runtime_config.session_directory, limit=limit)

    def filtered_recent_sessions(self, filter_text: str = '', *, limit: int = 10) -> tuple[AgentSessionSummary, ...]:
        sessions = self.recent_sessions(limit=max(limit * 5, limit, 10))
        query = filter_text.strip().casefold()
        if query:
            sessions = tuple(
                session
                for session in sessions
                if query in ' '.join(
                    [session.session_id, session.model, session.cwd, session.preview]
                ).casefold()
            )
        return sessions[:max(limit, 0)]

    def resume_recent_session(self, selector: str) -> str | None:
        selector = selector.strip()
        if not selector:
            return None
        if selector.isdigit():
            index = int(selector) - 1
            sessions = (
                self.last_session_picker
                if self.last_session_picker is not None
                else self.recent_sessions(limit=max(index + 1, 10))
            )
            if 0 <= index < len(sessions):
                self.resume_session(sessions[index].session_id)
                return sessions[index].session_id
            return None
        self.resume_session(selector)
        return self.session_id

    def format_recent_sessions(self, *, limit: int = 10, filter_text: str = '') -> str:
        sessions = self.filtered_recent_sessions(filter_text, limit=limit)
        self.last_session_picker = sessions
        self.last_session_filter = filter_text.strip()
        if not sessions:
            if self.last_session_filter:
                return f'No saved sessions found matching {self.last_session_filter!r}.'
            return 'No saved sessions found.'
        lines = []
        if self.last_session_filter:
            lines.append(f'Filter: {self.last_session_filter}')
        for index, session in enumerate(sessions, start=1):
            preview = f' | {session.preview}' if session.preview else ''
            lines.append(
                f'{index}. {session.session_id} | turns={session.turns} tools={session.tool_calls} '
                f'tokens={session.total_tokens} | model={session.model}{preview}'
            )
        return '\n'.join(lines)

    def context_report(self, prompt: str | None = None) -> str:
        return self._build_agent().render_context_report(prompt)

    def permissions_report(self) -> str:
        return self._build_agent().render_permissions_report()

    def tools_report(self) -> str:
        return self._build_agent().render_tools_report()

    def agents_report(self) -> str:
        delegate_model = self.runtime_config.delegate_model or self.model_config.model
        delegate_base_url = self.runtime_config.delegate_base_url or self.model_config.base_url
        lines = [
            '# Agents',
            '',
            f'- Main model: {self.model_config.model}',
            f'- Delegate model: {delegate_model}',
            f'- Delegate backend: {delegate_base_url}',
            f'- Current session: {self.session_id or "<new>"}',
            f'- Max delegated tasks: {self.runtime_config.budget_config.max_delegated_tasks or "unlimited"}',
        ]
        if self.last_result is not None:
            delegates = [
                event for event in self.last_result.events
                if str(event.get('type')) in {'delegate_subtask_result', 'delegate_group_result'}
            ]
            lines.append(f'- Last run delegation events: {len(delegates)}')
            for event in delegates[-5:]:
                label = event.get('label') or event.get('group_id') or event.get('index') or 'delegate'
                stop = event.get('stop_reason') or event.get('group_status') or 'done'
                child_model = event.get('child_model') or delegate_model
                child_base_url = event.get('child_base_url') or delegate_base_url
                lines.append(f'  - {label}: {stop} via {child_model} @ {child_base_url}')
        return '\n'.join(lines)

    def diff_report(self) -> str:
        try:
            inside = subprocess.run(
                ['git', 'rev-parse', '--is-inside-work-tree'],
                cwd=self.runtime_config.cwd,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            return f'# Diff\n\nUnable to run git: {exc}'
        if inside.returncode != 0 or inside.stdout.strip() != 'true':
            return '# Diff\n\nCurrent workspace is not inside a git worktree.'
        status = subprocess.run(
            ['git', 'status', '--short'],
            cwd=self.runtime_config.cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        unstaged = subprocess.run(
            ['git', 'diff', '--stat', '--patch', '--'],
            cwd=self.runtime_config.cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        staged = subprocess.run(
            ['git', 'diff', '--cached', '--stat', '--patch', '--'],
            cwd=self.runtime_config.cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        if status.returncode != 0 or unstaged.returncode != 0 or staged.returncode != 0:
            error = (status.stderr or unstaged.stderr or staged.stderr or 'git diff failed').strip()
            return f'# Diff\n\n{error}'
        untracked_patch = self._untracked_diff_report(status.stdout)
        if not status.stdout.strip() and not unstaged.stdout.strip() and not staged.stdout.strip() and not untracked_patch.strip():
            return '# Diff\n\nNo local changes.'
        parts = ['# Diff']
        if status.stdout.strip():
            parts.append('## Changed Files')
            parts.append(f'```text\n{status.stdout.strip()}\n```')
        if unstaged.stdout.strip():
            parts.append('## Unstaged Patch')
            parts.append(f'```diff\n{unstaged.stdout.rstrip()}\n```')
        if staged.stdout.strip():
            parts.append('## Staged Patch')
            parts.append(f'```diff\n{staged.stdout.rstrip()}\n```')
        if untracked_patch.strip():
            parts.append('## Untracked Files')
            parts.append(f'```diff\n{untracked_patch.rstrip()}\n```')
        return '\n\n'.join(parts)

    def _untracked_diff_report(self, status_stdout: str) -> str:
        chunks: list[str] = []
        for line in status_stdout.splitlines():
            if not line.startswith('?? '):
                continue
            relative_path = line[3:].strip()
            path = (self.runtime_config.cwd / relative_path).resolve()
            try:
                path.relative_to(self.runtime_config.cwd.resolve())
            except ValueError:
                continue
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                chunks.append(f'diff --git a/{relative_path} b/{relative_path}\nnew file mode 100644\n(Binary file not shown)')
                continue
            chunks.append(
                f'diff --git a/{relative_path} b/{relative_path}\n'
                'new file mode 100644\n'
                '--- /dev/null\n'
                f'+++ b/{relative_path}\n'
                + ''.join(f'+{line}\n' for line in content.splitlines())
            )
        return '\n'.join(chunks)

    def toggle_write(self) -> None:
        self._set_permissions(
            replace(
                self.runtime_config.permissions,
                allow_file_write=not self.runtime_config.permissions.allow_file_write,
            )
        )

    def toggle_shell(self) -> None:
        self._set_permissions(
            replace(
                self.runtime_config.permissions,
                allow_shell_commands=not self.runtime_config.permissions.allow_shell_commands,
            )
        )

    def toggle_unsafe(self) -> None:
        self._set_permissions(
            replace(
                self.runtime_config.permissions,
                allow_destructive_shell_commands=(
                    not self.runtime_config.permissions.allow_destructive_shell_commands
                ),
            )
        )

    def start_voice_recording(self) -> VoiceRecording:
        service = self.voice_input_service or VoiceInputService()
        return service.start_recording()

    def capture_voice_prompt(self) -> str:
        service = self.voice_input_service or VoiceInputService()
        return service.capture_and_transcribe()

    def status_lines(self) -> list[str]:
        permissions = self.runtime_config.permissions
        return [
            f'status={self.status}',
            f'session_id={self.session_id or "<new>"}',
            f'cwd={self.runtime_config.cwd}',
            f'model={self.model_config.model}',
            f'base_url={self.model_config.base_url}',
            f'stream={self.runtime_config.stream_model_responses}',
            f'write={permissions.allow_file_write}',
            f'shell={permissions.allow_shell_commands}',
            f'unsafe={permissions.allow_destructive_shell_commands}',
        ]

    def _set_permissions(self, permissions: AgentPermissions) -> None:
        self.runtime_config = replace(self.runtime_config, permissions=permissions)

    def _load_session(self, session_id: str) -> StoredAgentSession:
        return load_agent_session(session_id, directory=self.runtime_config.session_directory)

    def _build_agent(self) -> LocalCodingAgent:
        if self.agent_factory is not None:
            return self.agent_factory(self.model_config, self.runtime_config)
        return LocalCodingAgent(
            model_config=self.model_config,
            runtime_config=self.runtime_config,
            custom_system_prompt=self.custom_system_prompt,
            append_system_prompt=self.append_system_prompt,
            override_system_prompt=self.override_system_prompt,
        )


def _local_command_help() -> str:
    return '\n'.join(
        [
            'Commands:',
            '/status  show session/config',
            '/tools  show registered tools and permission state',
            '/permissions  show active permission mode',
            '/context  show estimated session context usage',
            '/diff  show changed files and patch for the current git workspace',
            '/agents  show main/delegate model routing and delegation status',
            '/find <text>  search the visible transcript',
            '/sessions [filter] or /history [filter]  list recent saved sessions',
            '/new  start a new session',
            '/resume <session-or-number>  resume a saved session',
            '/copy or /copy-last  copy the last user/assistant turn',
            '/export  export the transcript to markdown',
            '/write  toggle file write tools',
            '/shell  toggle shell commands',
            '/unsafe  toggle destructive shell commands',
            '/quit  exit',
        ]
    )

def render_plain_event(event: dict[str, object]) -> str:
    event_type = str(event.get('type', 'event'))
    if event_type in {'content_delta', 'message_delta'}:
        return f'model: {event.get("delta") or event.get("text") or ""}'
    if event_type == 'tool_start':
        return f'tool_start: {event.get("tool_name")} ({event.get("tool_call_id")})'
    if event_type == 'tool_delta':
        stream = event.get('stream') or 'tool'
        return f'tool_delta[{stream}]: {event.get("delta", "")}'
    if event_type == 'tool_result':
        return f'tool_result: {event.get("tool_name")} ok={event.get("ok")}'
    if event_type == 'delegate_subtask_result':
        return (
            'delegate_subtask: '
            f'{event.get("label") or event.get("index")} '
            f'session={event.get("session_id")} stop={event.get("stop_reason")}'
        )
    if event_type == 'delegate_group_result':
        return (
            'delegate_group: '
            f'{event.get("group_id")} status={event.get("group_status")} '
            f'completed={event.get("completed_children")} failed={event.get("failed_children")}'
        )
    if event_type == 'runtime_summary':
        return 'runtime_summary'
    return event_type


def run_tui(
    *,
    model_config: ModelConfig,
    runtime_config: AgentRuntimeConfig,
    custom_system_prompt: str | None = None,
    append_system_prompt: str | None = None,
    override_system_prompt: str | None = None,
) -> int:
    controller = TuiController(
        model_config=model_config,
        runtime_config=runtime_config,
        custom_system_prompt=custom_system_prompt,
        append_system_prompt=append_system_prompt,
        override_system_prompt=override_system_prompt,
    )
    import sys

    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            from .textual_tui_app import run_textual_tui
        except Exception:
            return _run_curses_tui(controller)
        return run_textual_tui(controller)

    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.prompt import Prompt
        from rich.table import Table
    except ImportError:
        return _run_plain_prompt(controller)

    console = Console()
    console.print(
        Panel(
            '*** CLAWEDCODE TUI MODE ***\n'
            'This is the new service-aware Rich TUI, not the old clawed REPL.\n'
            'Commands: /help, /status, /tools, /permissions, /context, /agents, /diff, '
            '/sessions filter, /new, /resume <session-or-number>, /copy-last, /export, '
            '/write, /shell, /unsafe, /quit',
            title='ClawedCode TUI',
        )
    )
    _print_status_panel(console, Panel, controller)

    while True:
        try:
            prompt = Prompt.ask('[bold green]clawed-tui[/]').strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        if not prompt:
            continue
        handled = _handle_rich_command(console, Panel, controller, prompt)
        if handled == 'quit':
            return 0
        if handled == 'handled':
            continue
        try:
            with console.status('Running local agent...', spinner='dots'):
                result = controller.run_prompt(prompt)
        except Exception as exc:  # pragma: no cover - terminal safety net
            console.print(Panel(str(exc), title='Error', border_style='red'))
            continue
        _print_result(console, Markdown, Panel, Table, controller, result)



class _TranscriptLine:
    def __init__(self, role: str, text: str) -> None:
        self.role = role
        self.text = text


def _run_curses_tui(controller: TuiController) -> int:
    return curses.wrapper(_CursesTui(controller).run)


class _CursesTui:
    def __init__(self, controller: TuiController) -> None:
        self.controller = controller
        self.lines: list[_TranscriptLine] = [
            _TranscriptLine('system', 'CLAWEDCODE TUI MODE'),
            _TranscriptLine('system', 'Full-screen terminal UI. User input stays in the bottom bar.'),
            _TranscriptLine('system', _local_command_help()),
            _TranscriptLine('system', 'Scroll: Up/Down, PageUp/PageDown, Home/End'),
        ]
        self.input_text = ''
        self.scroll = 0
        self.status = 'ready'

    def run(self, stdscr) -> int:
        curses.curs_set(1)
        curses.use_default_colors()
        self._init_colors()
        stdscr.keypad(True)
        # Keep terminal text selection/copy working. Mouse capture makes
        # GNOME Terminal send drag events to curses instead of selecting text.
        while True:
            self._draw(stdscr)
            key = stdscr.getch()
            if key in (curses.KEY_RESIZE,):
                continue
            if key in (10, 13):
                prompt = self.input_text.strip()
                self.input_text = ''
                if not prompt:
                    continue
                action = self._handle_command(prompt)
                if action == 'quit':
                    return 0
                if action == 'handled':
                    continue
                self._append('user', prompt)
                self.status = 'running'
                self._draw(stdscr)
                try:
                    result = self.controller.run_prompt(prompt)
                except Exception as exc:  # pragma: no cover - interactive safety net
                    self._append('system', f'error: {exc}')
                else:
                    assistant_text, tool_summary, delegate_summary = _format_result_for_display(result)
                    if assistant_text:
                        self._append('assistant', assistant_text)
                    if tool_summary:
                        self._append('system', f'Tool activity: {tool_summary}')
                    if delegate_summary:
                        self._append('system', f'Delegation: {delegate_summary}')
                    self._append(
                        'system',
                        f'session={result.session_id or self.controller.session_id or "<new>"} '
                        f'stop={result.stop_reason or "completed"} turns={result.turns} '
                        f'tools={result.tool_calls} tokens={result.usage.total_tokens}',
                    )
                self.status = 'ready'
                continue
            if key in (curses.KEY_BACKSPACE, 127, 8):
                self.input_text = self.input_text[:-1]
                continue
            if key == curses.KEY_DC:
                self.input_text = ''
                continue
            if key == curses.KEY_PPAGE:
                self._scroll_by(8, stdscr)
                continue
            if key == curses.KEY_NPAGE:
                self._scroll_by(-8, stdscr)
                continue
            if key == curses.KEY_UP:
                self._scroll_by(1, stdscr)
                continue
            if key == curses.KEY_DOWN:
                self._scroll_by(-1, stdscr)
                continue
            if key == curses.KEY_HOME:
                self._scroll_to_top(stdscr)
                continue
            if key == curses.KEY_END:
                self.scroll = 0
                continue
            if 0 <= key < 256:
                ch = chr(key)
                if ch.isprintable():
                    self.input_text += ch
        return 0

    def _init_colors(self) -> None:
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, -1)    # user
        curses.init_pair(2, curses.COLOR_GREEN, -1)   # assistant
        curses.init_pair(3, curses.COLOR_YELLOW, -1)  # system
        curses.init_pair(4, curses.COLOR_MAGENTA, -1) # tool
        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_WHITE) # input

    def _handle_command(self, prompt: str) -> str | None:
        command = prompt.strip()
        if command in {'/quit', ':quit', '/q', ':q', 'quit', 'exit'}:
            return 'quit'
        if command in {'/help', ':help', 'help'}:
            self._append('system', _local_command_help())
            return 'handled'
        if command in {'/status', ':status'}:
            for line in self.controller.status_lines():
                self._append('system', line)
            return 'handled'
        if command in {'/tools', ':tools'}:
            self._append('system', self.controller.tools_report())
            return 'handled'
        if command in {'/permissions', ':permissions'}:
            self._append('system', self.controller.permissions_report())
            return 'handled'
        if command in {'/context', ':context'}:
            self._append('system', self.controller.context_report())
            return 'handled'
        if command in {'/new', ':new'}:
            self.controller.start_new_session()
            self._append('system', 'started new session')
            return 'handled'
        if command.startswith('/resume ') or command.startswith(':resume '):
            self.controller.resume_session(command.split(maxsplit=1)[1])
            self._append('system', f'resuming session {self.controller.session_id}')
            return 'handled'
        if command in {'/write', ':write'}:
            self.controller.toggle_write()
            self._append('system', f'write={self.controller.runtime_config.permissions.allow_file_write}')
            return 'handled'
        if command in {'/shell', ':shell'}:
            self.controller.toggle_shell()
            self._append('system', f'shell={self.controller.runtime_config.permissions.allow_shell_commands}')
            return 'handled'
        if command in {'/unsafe', ':unsafe'}:
            self.controller.toggle_unsafe()
            self._append('system', f'unsafe={self.controller.runtime_config.permissions.allow_destructive_shell_commands}')
            return 'handled'
        return None

    def _append(self, role: str, text: str) -> None:
        self.lines.append(_TranscriptLine(role, text or ''))
        self.scroll = 0

    def _draw(self, stdscr) -> None:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height < 8 or width < 40:
            self._safe_addnstr(stdscr, 0, 0, 'Terminal too small for ClawedCode TUI', width - 1)
            stdscr.refresh()
            return

        # Curses raises ERR when writing the terminal's lower-right cell on
        # some terminals. Keep all writes one column short and pad visually.
        safe_width = max(1, width - 1)
        scroll_hint = f' | scroll +{self.scroll}' if self.scroll else ''
        header = (
            f' ClawedCode TUI | {self.status} | model={self.controller.model_config.model} '
            f'| session={self.controller.session_id or "<new>"}{scroll_hint} '
        )
        self._safe_addnstr(stdscr, 0, 0, header.ljust(width), safe_width, curses.A_REVERSE)

        body_height = height - 4
        rendered = self._render_lines(width)
        self.scroll = min(self.scroll, max(0, len(rendered) - body_height))
        if self.scroll:
            start = max(0, len(rendered) - body_height - self.scroll)
        else:
            start = max(0, len(rendered) - body_height)
        visible = rendered[start:start + body_height]
        for row, (role, text) in enumerate(visible, start=1):
            self._safe_addnstr(stdscr, row, 0, text.ljust(width), safe_width, self._role_attr(role))

        divider_y = height - 3
        self._safe_addnstr(stdscr, divider_y, 0, '-' * width, safe_width, curses.color_pair(5))
        prompt_label = ' USER PROMPT  (Enter sends, Up/Down/PageUp/PageDown scroll) '
        self._safe_addnstr(
            stdscr,
            height - 2,
            0,
            prompt_label.ljust(width),
            safe_width,
            curses.color_pair(5) | curses.A_BOLD,
        )
        input_line = self.input_text[-max(1, width - 2):]
        self._safe_addnstr(stdscr, height - 1, 0, input_line.ljust(width), safe_width, curses.color_pair(5))
        stdscr.move(height - 1, min(len(input_line), safe_width - 1))
        stdscr.refresh()

    def _safe_addnstr(self, stdscr, y: int, x: int, text: str, width: int, attr: int = 0) -> None:
        try:
            stdscr.addnstr(y, x, text, max(0, width), attr)
        except curses.error:
            pass

    def _scroll_by(self, amount: int, stdscr) -> None:
        self.scroll = min(self._max_scroll(stdscr), max(0, self.scroll + amount))

    def _scroll_to_top(self, stdscr) -> None:
        self.scroll = self._max_scroll(stdscr)

    def _max_scroll(self, stdscr) -> int:
        height, width = stdscr.getmaxyx()
        body_height = max(1, height - 4)
        return max(0, len(self._render_lines(width)) - body_height)

    def _render_lines(self, width: int) -> list[tuple[str, str]]:
        rendered: list[tuple[str, str]] = []
        label_width = len(self._role_label('assistant'))
        wrap_width = max(20, width - label_width - 1)
        for line in self.lines:
            label = self._role_label(line.role)
            continuation = self._continuation_label(label_width)
            first_line = True
            for physical_line in _markdown_display_lines(line.text):
                parts = _wrap_display_line(physical_line, wrap_width)
                for part in parts:
                    prefix = label if first_line else continuation
                    rendered.append((line.role, f'{prefix}{part}'))
                    first_line = False
        return rendered

    def _continuation_label(self, width: int) -> str:
        if width <= 2:
            return ' ' * width
        return f'{" " * (width - 2)}| '

    def _role_label(self, role: str) -> str:
        labels = {
            'user': 'USER      | ',
            'assistant': 'ASSISTANT | ',
            'system': 'SYSTEM    | ',
            'tool': 'TOOL      | ',
        }
        return labels.get(role, f'{role.upper()[:9]:<10}| ')

    def _role_attr(self, role: str) -> int:
        if role == 'user':
            return curses.color_pair(1) | curses.A_BOLD
        if role == 'assistant':
            return curses.color_pair(2)
        if role == 'tool':
            return curses.color_pair(4)
        return curses.color_pair(3)


def _format_result_for_display(result: AgentRunResult) -> tuple[str, str, str]:
    return (
        _normalize_assistant_text(result.final_output or ''),
        _summarize_event_group(result.events, prefix='tool'),
        _summarize_event_group(result.events, prefix='delegate'),
    )


def _summarize_event_group(events: tuple[dict[str, object], ...], *, prefix: str) -> str:
    items: list[str] = []
    for event in events:
        event_type = str(event.get('type', 'event'))
        if prefix == 'tool' and event_type == 'tool_result':
            name = str(event.get('tool_name') or 'tool')
            ok = event.get('ok')
            status = 'ok' if ok is True else 'failed' if ok is False else 'done'
            items.append(f'{name} {status}')
        elif prefix == 'delegate' and event_type == 'delegate_subtask_result':
            label = event.get('label') or event.get('index') or 'subtask'
            stop = event.get('stop_reason') or 'done'
            items.append(f'{label} {stop}')
        elif prefix == 'delegate' and event_type == 'delegate_group_result':
            group_id = event.get('group_id') or 'group'
            completed = event.get('completed_children')
            failed = event.get('failed_children')
            items.append(f'{group_id}: {completed} completed, {failed} failed')
    if not items:
        return ''
    shown = items[:5]
    suffix = f', +{len(items) - 5} more' if len(items) > 5 else ''
    return ', '.join(shown) + suffix


def _normalize_assistant_text(text: str) -> str:
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].strip().lower() in {'assistant', 'assistant:'}:
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    if _starts_with_empty_think_block(lines):
        lines = lines[3:]
        while lines and not lines[0].strip():
            lines.pop(0)
    return '\n'.join(lines).strip()


def _starts_with_empty_think_block(lines: list[str]) -> bool:
    return (
        len(lines) >= 3
        and lines[0].strip().lower() == '<think>'
        and not lines[1].strip()
        and lines[2].strip().lower() == '</think>'
    )


def _markdown_display_lines(text: str) -> list[str]:
    lines = text.splitlines() or ['']
    rendered: list[str] = []
    in_code_block = False
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            rendered.append(raw_line)
            continue
        if in_code_block:
            rendered.append(raw_line)
            continue
        rendered.append(_format_markdown_line(raw_line))
    return rendered


def _format_markdown_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ''
    if set(stripped) <= {'-'} and len(stripped) >= 3:
        return '-' * min(40, len(stripped))
    if stripped.startswith('#'):
        heading = stripped.lstrip('#').strip()
        return heading
    return line


def _wrap_display_line(line: str, width: int) -> list[str]:
    if not line:
        return ['']
    leading = len(line) - len(line.lstrip(' '))
    stripped = line[leading:]
    continuation = leading
    if stripped.startswith(('- ', '* ', '+ ')):
        continuation += 2
    else:
        marker_end = _ordered_list_marker_width(stripped)
        if marker_end:
            continuation += marker_end
    subsequent_indent = ' ' * min(continuation, max(0, width - 8))
    return textwrap.wrap(
        line,
        width=width,
        replace_whitespace=False,
        drop_whitespace=True,
        break_long_words=True,
        subsequent_indent=subsequent_indent,
    ) or ['']


def _ordered_list_marker_width(text: str) -> int:
    index = 0
    while index < len(text) and text[index].isdigit():
        index += 1
    if index == 0 or index + 1 >= len(text):
        return 0
    if text[index] in {'.', ')'} and text[index + 1] == ' ':
        return index + 2
    return 0

def _run_plain_prompt(controller: TuiController) -> int:
    print('*** CLAWEDCODE TUI MODE ***')
    print('This is the new service-aware TUI, not the old clawed REPL.')
    print(_local_command_help())
    for line in controller.status_lines():
        print(line)
    while True:
        try:
            prompt = input('clawed-tui> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        handled = _handle_plain_command(controller, prompt)
        if handled == 'quit':
            return 0
        if handled == 'handled':
            continue
        try:
            result = controller.run_prompt(prompt)
        except Exception as exc:  # pragma: no cover - terminal safety net
            print(f'error: {exc}')
            continue
        for event in result.events:
            print(render_plain_event(event))
        print(result.final_output)
        print(f'session_id={result.session_id or controller.session_id or "<new>"}')
        print(f'stop_reason={result.stop_reason or "completed"}')


def _handle_plain_command(controller: TuiController, prompt: str) -> str | None:
    command = prompt.strip()
    if command in {'/quit', ':quit', '/q', ':q', 'quit', 'exit'}:
        return 'quit'
    if command in {'/help', ':help', 'help'}:
        print(_local_command_help())
        return 'handled'
    if command in {'/status', ':status'}:
        for line in controller.status_lines():
            print(line)
        return 'handled'
    if command in {'/tools', ':tools'}:
        print(controller.tools_report())
        return 'handled'
    if command in {'/permissions', ':permissions'}:
        print(controller.permissions_report())
        return 'handled'
    if command in {'/context', ':context'}:
        print(controller.context_report())
        return 'handled'
    if command in {'/diff', ':diff'}:
        print(controller.diff_report())
        return 'handled'
    if command in {'/agents', ':agents'}:
        print(controller.agents_report())
        return 'handled'
    if command in {'/sessions', ':sessions', '/history', ':history'}:
        print(controller.format_recent_sessions())
        return 'handled'
    if command.startswith('/sessions ') or command.startswith(':sessions ') or command.startswith('/history ') or command.startswith(':history '):
        print(controller.format_recent_sessions(filter_text=command.split(maxsplit=1)[1]))
        return 'handled'
    if command in {'/copy', '/copy-turn', '/copy-last', ':copy', ':copy-turn', ':copy-last', '/export', ':export', '/find', ':find'} or command.startswith(('/find ', ':find ', '/search ', ':search ')):
        print('This command is available in the Textual TUI only.')
        return 'handled'
    if command in {'/new', ':new'}:
        controller.start_new_session()
        print('session reset')
        return 'handled'
    if command.startswith('/resume ') or command.startswith(':resume '):
        resumed = controller.resume_recent_session(command.split(maxsplit=1)[1])
        if resumed:
            print(f'resuming session_id={resumed}')
        else:
            print('No matching saved session. Use /sessions to list recent sessions.')
        return 'handled'
    if command in {'/write', ':write'}:
        controller.toggle_write()
        print(f'write={controller.runtime_config.permissions.allow_file_write}')
        return 'handled'
    if command in {'/shell', ':shell'}:
        controller.toggle_shell()
        print(f'shell={controller.runtime_config.permissions.allow_shell_commands}')
        return 'handled'
    if command in {'/unsafe', ':unsafe'}:
        controller.toggle_unsafe()
        print(f'unsafe={controller.runtime_config.permissions.allow_destructive_shell_commands}')
        return 'handled'
    return None


def _handle_rich_command(console, Panel, controller: TuiController, prompt: str) -> str | None:
    command = prompt.strip()
    if command in {'/quit', ':quit', '/q', ':q', 'quit', 'exit'}:
        return 'quit'
    if command in {'/help', ':help', 'help'}:
        console.print(Panel(_local_command_help(), title='Commands'))
        return 'handled'
    if command in {'/status', ':status'}:
        _print_status_panel(console, Panel, controller)
        return 'handled'
    if command in {'/tools', ':tools'}:
        console.print(Panel(controller.tools_report(), title='Tools'))
        return 'handled'
    if command in {'/permissions', ':permissions'}:
        console.print(Panel(controller.permissions_report(), title='Permissions'))
        return 'handled'
    if command in {'/context', ':context'}:
        console.print(Panel(controller.context_report(), title='Context'))
        return 'handled'
    if command in {'/diff', ':diff'}:
        console.print(Panel(controller.diff_report(), title='Diff'))
        return 'handled'
    if command in {'/agents', ':agents'}:
        console.print(Panel(controller.agents_report(), title='Agents'))
        return 'handled'
    if command in {'/sessions', ':sessions', '/history', ':history'}:
        console.print(Panel(controller.format_recent_sessions(), title='Sessions'))
        return 'handled'
    if command.startswith('/sessions ') or command.startswith(':sessions ') or command.startswith('/history ') or command.startswith(':history '):
        console.print(Panel(controller.format_recent_sessions(filter_text=command.split(maxsplit=1)[1]), title='Sessions'))
        return 'handled'
    if command in {'/copy', '/copy-turn', '/copy-last', ':copy', ':copy-turn', ':copy-last', '/export', ':export', '/find', ':find'} or command.startswith(('/find ', ':find ', '/search ', ':search ')):
        console.print(Panel('This command is available in the Textual TUI only.', title='Textual only'))
        return 'handled'
    if command in {'/new', ':new'}:
        controller.start_new_session()
        _print_status_panel(console, Panel, controller)
        return 'handled'
    if command.startswith('/resume ') or command.startswith(':resume '):
        resumed = controller.resume_recent_session(command.split(maxsplit=1)[1])
        if resumed:
            _print_status_panel(console, Panel, controller)
        else:
            console.print(Panel('No matching saved session. Use /sessions to list recent sessions.', title='Sessions'))
        return 'handled'
    if command in {'/write', ':write'}:
        controller.toggle_write()
        _print_status_panel(console, Panel, controller)
        return 'handled'
    if command in {'/shell', ':shell'}:
        controller.toggle_shell()
        _print_status_panel(console, Panel, controller)
        return 'handled'
    if command in {'/unsafe', ':unsafe'}:
        controller.toggle_unsafe()
        _print_status_panel(console, Panel, controller)
        return 'handled'
    return None


def _print_status_panel(console, Panel, controller: TuiController) -> None:
    console.print(Panel('\n'.join(controller.status_lines()), title='Status'))


def _print_result(console, Markdown, Panel, Table, controller: TuiController, result: AgentRunResult) -> None:
    assistant_text, tool_summary, delegate_summary = _format_result_for_display(result)
    if assistant_text:
        console.print(Panel(Markdown(assistant_text), title='Assistant'))
    if tool_summary:
        console.print(Panel(tool_summary, title='Tool activity', border_style='yellow'))
    if delegate_summary:
        console.print(Panel(delegate_summary, title='Delegation', border_style='magenta'))
    footer = [
        f'session_id={result.session_id or controller.session_id or "<new>"}',
        f'stop_reason={result.stop_reason or "completed"}',
        f'turns={result.turns}',
        f'tool_calls={result.tool_calls}',
        f'tokens={result.usage.total_tokens}',
    ]
    if result.session_path:
        footer.append(f'session_path={result.session_path}')
    console.print(Panel('\n'.join(footer), title='Run'))
