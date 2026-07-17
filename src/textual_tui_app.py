from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime

from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Static, TextArea

from .agent_types import AgentRunResult, AgentRuntimeConfig, ModelConfig
from .tui_app import TuiController, _local_command_help, render_plain_event
from .voice_input import VoiceRecording

DRAGON_ART = r'''
                       ⡄    ⣠⢀⣀⣀⣤⣠
                       ⣿⡀⢀⣤⣾⠃⠉⠉⣼⡇⢐⠲⠦⣄⡀
                       ⢸⣿⣜⠿⣏ ⢀⣼⣿ ⠛⠳⣦⣀⢤⡀    ⢠ ⠾⠂
          ⠰⣏⡇    ⡀⣀   ⠲⣌⢿⣏⢻⣿⣿⢉⣿⢏⡴⣢⡀⠈⠻⡎⠳⡄   ⠈
                 ⠇⠟   ⢀⡌⢰⣿⣿⣿⡥⢈⠁⠈⣌⣿⣿⣆ ⢸⢰⡌      ⣠⠖
    ⢀⡀      ⠘⡆  ⣠⣤⡀⣀⣄⣴⣮⣷⣾⢿⡉⣿⡟⣻⢿⣶⣏⣿⣿⣿⡇⠈⠈⡇ ⢠⡄⢀⡴⠟⠁
    ⠈⠇⠐⠃      ⣀ ⠁⣼⣿⣁⣾⣿⣿⣿⣿⣷⡻⢻⡋⣱⣿⣿⣿⣿⣿⣿⣇⣀⣀⣀⡀⠈⠁⣸⠇    ⠓  ⢠
       ⢠⡄    ⢰⣏⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇  ⢿⡀  ⠘⠃   ⠛
  ⢀      ⢀⣀   ⠙⢿⣧⡤⣽⣤⣬⣤⣤⡄⠠⡄ ⢠⡤⠤⣠⣤⣤      ⢸⡇  ⠈⢿⣠⡀    ⣲
  ⠘      ⠸⠼⠃  ⢀⣸⡇⠆⠶⠶⠴⠲⠦ ⠆⠰⠂⠲⠆⠆       ⢀⣤⣾⣿⣳⡆ ⠈⢿⡟    ⡀  ⢐⡆
⢀⡀          ⣠⣴⡿⢻⡇⣤⣀⡀⣤⣀⡀⢠⣀⣠⣠⢀⣀⡀ ⡄       ⢹⡏⡽⠃  ⢸⡇   ⠸⠁
⠈⠁   ⢰⡲⡄   ⣸⣯⡟⠁⣸⡇                      ⢸⣇⡀  ⣠⣞⢣⠆ ⡀
    ⢠⠈⠁⠁  ⢰⣿⣿⠃⢸⣿⡇⠃                   ⠰⣶⣼⣟⡿⠿⠿⢟⣡⠏  ⠁   ⠰⠃
    ⠘    ⡆⢻⣿⣿⣄⠈⢿⡇⠔⠤⠤⠠⠤⠤⠄⠴⠴ ⠤ ⠄    ⣄ ⣿⣦⡈⠻⡿⣷⠲⠞⠋⠁    ⢀⡄
          ⠘⡿⣿⣿⣦⠸⠷⠶⠶⠶⠶⠶⠶⠶⠶⠶⢶⣶⣶⣶⡶⠶⠶⢾⣿⡶⠞⠻⢷⣼⠏⠁ ⡀⡀     ⠈⠁
       ⠰⣏⡇ ⠙⢌⠻⣿⣿⣿⣿⣶⣶⣶⣶⣶⠖⢀⣴⣿⣿⡿⠋⠛⠉ ⠈⠳⠶⠖ ⠶⠏  ⠈⠗⠃
             ⠑⢤⣈⡉⠉⠛⠛⠛⠉⢀⣤⣾⠿⠛⠋  ⢿⣹       ⠠⡆       ⠈⠁
                 ⠈   ⠘⠉⠁
        ⣠⣤⣤⣤⣠⡄   ⣠⣤⣤⣄⢠⣄⢀⣤⡀⣠⡄  ⣠⣤⣤⣤⢄⣤⣤⣤⡀⢤⣤⣤⣄ ⣤⣤⣤⣤⡄
       ⢸⣿⠉⠉⠁⣿⡇  ⢰⣿⢉⣭⣿⡟⣿⣼⣿⣧⣿⠃ ⢸⣿⠉⠉⠁⣿⡏⠉⢹⣿⣶⡏⠉⣿⡇⣿⣯⣿⡏
       ⠘⠿⠿⠿⠷⠛⠿⠿⠿⠾⠿⠉⠉⠿⠇⠹⠿⠈⠿⠏  ⠈⠻⠿⠿⠷⠝⠿⠿⠿⠋⠿⠿⠿⠿⠁⠿⠿⠿⠷⠄
'''.strip('\n')



PROMPT_MIN_HEIGHT = 1
PROMPT_MAX_HEIGHT = 6
PROMPT_CHROME_HEIGHT = 2
PROMPT_WRAP_FALLBACK_WIDTH = 80

_THEME_PALETTES = {
    'codex': {
        'screen': '#170719', 'panel': '#211026', 'panel_2': '#2d1533', 'text': '#f3eef7',
        'muted': '#cbbbd2', 'status_bg': '#241129', 'status_fg': '#f4edf7',
        'prompt_bg': '#28132e', 'prompt_fg': '#f7f1fb', 'prompt_label_bg': '#211026',
        'prompt_label_fg': '#f4edf7', 'input_bg': '#120816', 'input_fg': '#f8f5fb',
        'accent': '#b590ff', 'user': '#8bdcff', 'assistant': '#4ee89a',
        'system': '#e6b45c', 'tool': '#d89cff', 'border': '#6e4778',
    },
    'dragon': {
        'screen': '#080f0c', 'panel': '#0f2119', 'panel_2': '#173326', 'text': '#e9fff4',
        'muted': '#b9dcc8', 'status_bg': '#10251b', 'status_fg': '#eafff4',
        'prompt_bg': '#10251b', 'prompt_fg': '#ecfff5', 'prompt_label_bg': '#0b1a13',
        'prompt_label_fg': '#dfffee', 'input_bg': '#07120d', 'input_fg': '#f1fff7',
        'accent': '#36e695', 'user': '#77d7ff', 'assistant': '#36e695',
        'system': '#f0b95b', 'tool': '#c79cff', 'border': '#2f8f62',
    },
    'light': {
        'screen': '#f6f1e8', 'panel': '#ede4d8', 'panel_2': '#e3d8ca', 'text': '#211b18',
        'muted': '#665b54', 'status_bg': '#e9dfd2', 'status_fg': '#211b18',
        'prompt_bg': '#eadfce', 'prompt_fg': '#211b18', 'prompt_label_bg': '#ded1c1',
        'prompt_label_fg': '#211b18', 'input_bg': '#fffaf2', 'input_fg': '#211b18',
        'accent': '#6e4fb0', 'user': '#006b8f', 'assistant': '#087a43',
        'system': '#9a5f00', 'tool': '#7d4aa3', 'border': '#b7a899',
    },
}


def _theme_css(theme_name: str | None = None) -> str:
    palette = _THEME_PALETTES.get((theme_name or 'codex').strip().lower(), _THEME_PALETTES['codex'])
    return f'''
    Screen {{
        background: {palette['screen']};
        color: {palette['text']};
    }}

    Header {{
        background: {palette['panel_2']};
        color: {palette['text']};
    }}

    Footer {{
        background: {palette['panel_2']};
        color: {palette['muted']};
    }}

    #root {{
        height: 100%;
        layout: vertical;
        background: {palette['screen']};
    }}

    #status-bar {{
        height: 1;
        background: {palette['status_bg']};
        color: {palette['status_fg']};
        padding: 0 1;
    }}

    #activity-bar {{
        height: 1;
        background: {palette['panel']};
        color: {palette['muted']};
        padding: 0 1;
    }}

    #main {{
        height: 1fr;
        min-height: 5;
        background: {palette['screen']};
    }}

    #transcript {{
        height: 100%;
        padding: 0 1;
        background: {palette['screen']};
        scrollbar-background: {palette['screen']};
        scrollbar-color: {palette['border']};
        scrollbar-size: 1 1;
    }}

    #prompt-row {{
        height: 3;
        min-height: 3;
        max-height: 8;
        background: {palette['prompt_bg']};
        color: {palette['prompt_fg']};
        border-top: solid {palette['border']};
        padding: 0 1;
    }}

    #prompt-label {{
        width: 14;
        content-align: left top;
        text-style: bold;
        color: {palette['prompt_label_fg']};
        background: {palette['prompt_label_bg']};
        padding-top: 0;
    }}

    #prompt {{
        width: 1fr;
        height: 1;
        min-height: 1;
        max-height: 6;
        background: {palette['input_bg']};
        color: {palette['input_fg']};
        border: none;
    }}

    #hint-bar {{
        height: 1;
        background: {palette['panel_2']};
        color: {palette['muted']};
        padding: 0 1;
    }}

    .message {{
        margin: 1 0 0 0;
        padding: 0;
        layout: horizontal;
        height: auto;
    }}

    .role-label {{
        width: 12;
        min-width: 12;
        text-style: bold;
    }}

    .message-body {{
        width: 1fr;
        height: auto;
    }}

    .user .role-label {{ color: {palette['user']}; }}
    .assistant .role-label {{ color: {palette['assistant']}; }}
    .system .role-label {{ color: {palette['system']}; }}
    .tool .role-label {{ color: {palette['tool']}; }}

    .user .message-body {{ color: {palette['text']}; }}
    .assistant .message-body {{ color: {palette['text']}; }}
    .system .message-body {{ color: {palette['system']}; }}
    .tool .message-body {{ color: {palette['tool']}; }}

    .splash {{
        margin: 1 0;
        padding: 1 2;
        border: solid {palette['accent']};
        color: {palette['accent']};
        background: {palette['panel']};
        height: auto;
    }}
    '''


def _prompt_content_height(text: str, *, width: int | None) -> int:
    usable_width = max((width or PROMPT_WRAP_FALLBACK_WIDTH) - 1, 20)
    rows = 0
    for line in (text or '').split('\n'):
        rows += max(1, (len(line) // usable_width) + 1)
    return min(max(rows, PROMPT_MIN_HEIGHT), PROMPT_MAX_HEIGHT)

@dataclass(frozen=True)
class TranscriptEntry:
    role: str
    body: str


class PromptTextArea(TextArea):
    BINDINGS = [
        Binding('enter', 'submit_prompt', 'Send', show=False, priority=True),
        Binding('shift+enter,ctrl+j', 'insert_newline', 'Newline', show=False, priority=True),
        *TextArea.BINDINGS,
    ]

    class Submitted(Message):
        def __init__(self, prompt: 'PromptTextArea', value: str) -> None:
            super().__init__()
            self.prompt = prompt
            self.value = value

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__(
            '',
            soft_wrap=True,
            show_line_numbers=False,
            tab_behavior='focus',
            **kwargs,
        )

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, text: str) -> None:
        self.load_text(text)

    def action_submit_prompt(self) -> None:
        self.post_message(self.Submitted(self, self.text))

    def action_insert_newline(self) -> None:
        self.insert('\n')


class MessageBlock(Static):
    def __init__(self, entry: TranscriptEntry) -> None:
        super().__init__(classes=f'message {entry.role}')
        self.entry = entry
        self.body_widget: Static | None = None

    def compose(self) -> ComposeResult:
        yield Static(self.entry.role.upper(), classes='role-label')
        self.body_widget = Static(self._render_body(self.entry.body), classes=self._body_classes())
        yield self.body_widget

    def update_body(self, body: str) -> None:
        self.entry = TranscriptEntry(role=self.entry.role, body=body)
        if self.body_widget is not None:
            self.body_widget.update(self._render_body(body))

    def _render_body(self, body: str):  # noqa: ANN202
        if self.entry.role == 'assistant':
            return RichMarkdown(body or '')
        if self.entry.role == 'tool':
            return Syntax(body or '', 'text', word_wrap=True)
        return Text(body or '', overflow='fold')

    def _body_classes(self) -> str:
        if self.entry.role == 'assistant':
            return 'message-body markdown-body'
        if self.entry.role == 'tool':
            return 'message-body tool-body'
        return 'message-body'


class ClawedTextualApp(App[None]):
    CSS = _theme_css(os.environ.get('CLAWED_TUI_THEME'))

    BINDINGS = [
        ('ctrl+q', 'quit', 'Quit'),
        ('ctrl+n', 'new_session', 'New'),
        ('ctrl+s', 'show_status', 'Status'),
        ('ctrl+w', 'toggle_write', 'Write'),
        ('ctrl+b', 'toggle_shell', 'Shell'),
        ('ctrl+u', 'toggle_unsafe', 'Unsafe'),
        ('f4', 'voice_input', 'Voice'),
        ('alt+up', 'prompt_history_previous', 'Prev Prompt'),
        ('alt+down', 'prompt_history_next', 'Next Prompt'),
        Binding('ctrl+f', 'focus_find', 'Find', priority=True),
        ('ctrl+y', 'copy_last_turn', 'Copy'),
        ('ctrl+e', 'export_transcript', 'Export'),
        ('pageup', 'page_up', 'PgUp'),
        ('pagedown', 'page_down', 'PgDn'),
        ('home', 'scroll_home', 'Top'),
        ('end', 'scroll_end', 'Bottom'),
    ]

    TITLE = 'ClawedCode TUI'
    SUB_TITLE = 'Coding agent TUI'

    def __init__(self, controller: TuiController) -> None:
        super().__init__()
        self.controller = controller
        self.busy = False
        self.busy_started_at: float | None = None
        self.live_assistant_block: MessageBlock | None = None
        self.live_assistant_text = ''
        self.live_tool_counts: dict[str, int] = {}
        self.voice_recording: VoiceRecording | None = None
        self.voice_started_at: float | None = None
        self.transcript_entries: list[TranscriptEntry] = []
        self.transcript_blocks: list[MessageBlock] = []
        self.prompt_history: list[str] = []
        self.prompt_history_index: int | None = None
        self.prompt_history_draft = ''
        self.transcript_search_query = ''
        self.transcript_search_match_index: int | None = None
        self.transcript_search_matches: list[int] = []
        self.pending_permission_confirmation: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id='root'):
            yield Static('', id='status-bar')
            yield Static('idle', id='activity-bar')
            with Container(id='main'):
                with VerticalScroll(id='transcript'):
                    yield Static(self._splash_text(), classes='splash')
            with Horizontal(id='prompt-row'):
                yield Static('USER PROMPT', id='prompt-label')
                yield PromptTextArea(id='prompt')
            yield Static('Enter sends | Shift+Enter/Ctrl+J newline | Ctrl+F find | Alt+Up/Down history | F4 voice | /copy-last | /export | Ctrl+Q quit', id='hint-bar')
        yield Footer()

    def on_mount(self) -> None:
        prompt = self.query_one('#prompt', PromptTextArea)
        self._resize_prompt_for_content(prompt.value)
        prompt.focus()
        self._append_system('Textual UI loaded. Markdown rendering, scrollback, resize reflow, and keyboard navigation are active.')
        self.set_interval(1.0, self._refresh_running_status)
        self._update_status('ready')

    def on_text_area_changed(self, event) -> None:  # noqa: ANN001
        if getattr(event, 'text_area', None).id == 'prompt':
            self._resize_prompt_for_content(event.text_area.text)

    async def on_prompt_text_area_submitted(self, event: PromptTextArea.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt or self.busy:
            return
        event.prompt.value = ''
        self._resize_prompt_for_content('')
        command_result = self._handle_command(prompt)
        if command_result == 'quit':
            self.exit()
            return
        if command_result == 'handled':
            return
        self._remember_prompt(prompt)
        self._append_entry('user', prompt)
        await self._run_agent(prompt)

    async def _run_agent(self, prompt: str) -> None:
        self.busy = True
        self.busy_started_at = time.monotonic()
        self.live_assistant_block = None
        self.live_assistant_text = ''
        self.live_tool_counts = {}
        prompt_input = self.query_one('#prompt', PromptTextArea)
        prompt_input.disabled = True
        self._update_activity('model request started; waiting for stream...')
        self._append_system('Model request started. Waiting for stream...')
        self._update_status('running')
        def publish_event(event: dict[str, object]) -> None:
            self.call_from_thread(self._handle_live_event, event)

        try:
            result = await asyncio.to_thread(
                lambda: self.controller.run_prompt(prompt, event_callback=publish_event)
            )
        except Exception as exc:
            self._update_activity(f'error: {exc}')
            self._append_system(f'error: {exc}')
        else:
            self._append_result(result)
        finally:
            self.busy = False
            self.busy_started_at = None
            prompt_input.disabled = False
            prompt_input.focus()
            self._update_status('ready')

    def _append_result(self, result: AgentRunResult) -> None:
        tool_summary, delegate_summary = _summarize_run_events(result.events)
        control_summary = _summarize_control_events(result.events, stop_reason=result.stop_reason)
        assistant_body = _normalize_for_display(result.final_output or '')
        if assistant_body:
            if self.live_assistant_block is not None:
                self.live_assistant_block.update_body(assistant_body)
                self._replace_transcript_entry(self.live_assistant_block, assistant_body)
            else:
                self._append_entry('assistant', assistant_body)
        elif result.tool_calls or tool_summary or delegate_summary:
            self._append_entry('system', 'No final assistant response was produced after tool activity.')
        if tool_summary:
            self._append_entry('system', f'Tool activity: {tool_summary}')
        if delegate_summary:
            self._append_entry('system', f'Delegation: {delegate_summary}')
        if control_summary:
            self._append_entry('system', f'Control: {control_summary}')
        completion_state = result.stop_reason or 'completed'
        self._update_activity(
            f'completed: stop={completion_state} turns={result.turns} tools={result.tool_calls} tokens={result.usage.total_tokens}'
        )
        self._append_system(
            f'session={result.session_id or self.controller.session_id or "<new>"} '
            f'stop={completion_state} turns={result.turns} '
            f'tools={result.tool_calls} tokens={result.usage.total_tokens}'
        )

    def _handle_command(self, prompt: str) -> str | None:
        command = prompt.strip()
        if command in {'/quit', ':quit', '/q', ':q', 'quit', 'exit'}:
            return 'quit'
        if command in {'/help', ':help', 'help'}:
            self._append_system(_local_command_help())
            return 'handled'
        if command in {'/status', ':status'}:
            self._append_system('\n'.join(self.controller.status_lines()))
            return 'handled'
        if command in {'/tools', ':tools'}:
            self._append_system(self.controller.tools_report())
            return 'handled'
        if command in {'/permissions', ':permissions'}:
            self._append_system(self.controller.permissions_report())
            return 'handled'
        if command in {'/context', ':context'}:
            self._append_system(self.controller.context_report())
            return 'handled'
        if command in {'/diff', ':diff'}:
            self._append_system(self.controller.diff_report())
            return 'handled'
        if command in {'/agents', ':agents'}:
            self._append_system(self.controller.agents_report())
            return 'handled'
        if command == '/find' or command == ':find' or command == '/search' or command == ':search':
            self._append_system('Usage: /find <text>')
            return 'handled'
        if command.startswith('/find ') or command.startswith(':find ') or command.startswith('/search ') or command.startswith(':search '):
            self._find_transcript(command.split(maxsplit=1)[1])
            return 'handled'
        if command in {'/sessions', ':sessions', '/history', ':history'}:
            self._append_system(self.controller.format_recent_sessions())
            return 'handled'
        if command.startswith('/sessions ') or command.startswith(':sessions ') or command.startswith('/history ') or command.startswith(':history '):
            self._append_system(self.controller.format_recent_sessions(filter_text=command.split(maxsplit=1)[1]))
            return 'handled'
        if command in {'/copy', '/copy-turn', '/copy-last', ':copy', ':copy-turn', ':copy-last'}:
            self.action_copy_last_turn()
            return 'handled'
        if command in {'/export', ':export'}:
            self.action_export_transcript()
            return 'handled'
        if command in {'/new', ':new'}:
            self.controller.start_new_session()
            self._clear_transcript_entries()
            self._append_system('started new session')
            self._update_status('ready')
            return 'handled'
        if command.startswith('/resume ') or command.startswith(':resume '):
            resumed_session = self.controller.resume_recent_session(command.split(maxsplit=1)[1])
            if resumed_session:
                self._clear_transcript_entries()
                self._append_system(f'resuming session {resumed_session}')
            else:
                self._append_system('No matching saved session. Use /sessions to list recent sessions.')
            self._update_status('ready')
            return 'handled'
        if command in {'/write', ':write'}:
            self.controller.toggle_write()
            self._append_system(f'write={self.controller.runtime_config.permissions.allow_file_write}')
            self._update_status('ready')
            return 'handled'
        if command in {'/shell', ':shell'}:
            self._request_or_toggle_shell()
            return 'handled'
        if command in {'/unsafe', ':unsafe'}:
            self._request_or_toggle_unsafe()
            return 'handled'
        if command in {'/confirm shell', ':confirm shell'}:
            self._confirm_permission('shell')
            return 'handled'
        if command in {'/confirm unsafe', ':confirm unsafe'}:
            self._confirm_permission('unsafe')
            return 'handled'
        return None

    def _remember_prompt(self, prompt: str) -> None:
        if self.prompt_history and self.prompt_history[-1] == prompt:
            self.prompt_history_index = None
            self.prompt_history_draft = ''
            return
        self.prompt_history.append(prompt)
        self.prompt_history_index = None
        self.prompt_history_draft = ''

    def _set_prompt_value(self, text: str) -> None:
        prompt = self.query_one('#prompt', PromptTextArea)
        prompt.value = text
        self._resize_prompt_for_content(text)
        lines = text.split('\n')
        prompt.cursor_location = (len(lines) - 1, len(lines[-1]))
        prompt.focus()

    def _resize_prompt_for_content(self, text: str) -> None:
        prompt = self.query_one('#prompt', PromptTextArea)
        prompt_width = prompt.size.width or PROMPT_WRAP_FALLBACK_WIDTH
        height = _prompt_content_height(text, width=prompt_width)
        prompt.styles.height = height
        prompt.styles.min_height = height
        prompt_row = self.query_one('#prompt-row', Horizontal)
        prompt_row.styles.height = height + PROMPT_CHROME_HEIGHT
        prompt_row.styles.min_height = height + PROMPT_CHROME_HEIGHT

    def _append_system(self, text: str) -> None:
        self._append_entry('system', text)

    def _clear_transcript_entries(self) -> None:
        for block in list(self.transcript_blocks):
            block.remove()
        self.transcript_entries = []
        self.transcript_blocks = []
        self.transcript_search_query = ''
        self.transcript_search_match_index = None
        self.transcript_search_matches = []
        self.live_assistant_block = None
        self.live_assistant_text = ''
        self.live_tool_counts = {}

    def _append_entry(self, role: str, body: str) -> MessageBlock:
        transcript = self.query_one('#transcript', VerticalScroll)
        entry = TranscriptEntry(role=role, body=body or '')
        block = MessageBlock(entry)
        self.transcript_entries.append(entry)
        self.transcript_blocks.append(block)
        transcript.mount(block)
        self.call_after_refresh(transcript.scroll_end, animate=False)
        return block

    def _replace_transcript_entry(self, block: MessageBlock, body: str) -> None:
        try:
            index = self.transcript_blocks.index(block)
        except ValueError:
            return
        self.transcript_entries[index] = TranscriptEntry(role=block.entry.role, body=body or '')

    def _copy_text(self, text: str, label: str) -> None:
        if not text:
            self._append_system(f'{label}: nothing to copy')
            return
        try:
            self.copy_to_clipboard(text)
        except Exception as exc:  # pragma: no cover - backend depends on terminal/desktop
            path = self._write_copy_fallback(text, label)
            self._append_system(f'{label}: clipboard unavailable ({exc}); written: {path}')
            return
        self._append_system(f'{label} copied to clipboard')

    def _write_copy_fallback(self, text: str, label: str) -> Path:
        export_dir = self.controller.runtime_config.session_directory / 'exports'
        export_dir.mkdir(parents=True, exist_ok=True)
        slug = '-'.join(part for part in label.lower().split() if part) or 'copy'
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        path = export_dir / f'clawed-{slug}-{timestamp}.txt'
        path.write_text(text, encoding='utf-8')
        return path

    def _format_transcript_entries(self, entries: list[TranscriptEntry]) -> str:
        return '\n\n'.join(
            f'{entry.role.upper()}: {entry.body}'.rstrip()
            for entry in entries
            if entry.body.strip()
        )

    def _format_transcript_markdown(self) -> str:
        parts = ['# ClawedCode Transcript']
        for entry in self.transcript_entries:
            if not entry.body.strip():
                continue
            parts.append(f'## {entry.role.upper()}\n\n{entry.body}')
        return '\n\n'.join(parts).rstrip() + '\n'

    def _last_conversation_turn(self) -> list[TranscriptEntry]:
        selected: list[TranscriptEntry] = []
        for entry in reversed(self.transcript_entries):
            if entry.role not in {'user', 'assistant'}:
                if selected:
                    continue
                continue
            selected.append(entry)
            if entry.role == 'user' and len(selected) > 1:
                break
        return list(reversed(selected))

    def _find_transcript(self, query: str) -> None:
        query = query.strip()
        if not query:
            self._append_system('Usage: /find <text>')
            return
        matches = self._transcript_match_indices(query)
        if not matches:
            self.transcript_search_query = query
            self.transcript_search_matches = []
            self.transcript_search_match_index = None
            self._append_system(f'No transcript matches for \"{query}\"')
            return
        if query.casefold() != self.transcript_search_query.casefold() or matches != self.transcript_search_matches:
            self.transcript_search_query = query
            self.transcript_search_matches = matches
            self.transcript_search_match_index = 0
        else:
            current = self.transcript_search_match_index or 0
            self.transcript_search_match_index = (current + 1) % len(matches)
        self._show_transcript_match(query)

    def _transcript_match_indices(self, query: str) -> list[int]:
        lowered = query.casefold()
        return [
            index
            for index, entry in enumerate(self.transcript_entries)
            if entry.role in {'user', 'assistant', 'tool'} and lowered in entry.body.casefold()
        ]

    def _show_transcript_match(self, query: str) -> None:
        if self.transcript_search_match_index is None:
            return
        entry_index = self.transcript_search_matches[self.transcript_search_match_index]
        block = self.transcript_blocks[entry_index]
        try:
            block.scroll_visible(animate=False)
        except Exception:
            pass
        entry = self.transcript_entries[entry_index]
        self._append_system(
            f'Search {self.transcript_search_match_index + 1}/{len(self.transcript_search_matches)}: '
            f'{entry.role.upper()} contains \"{query}\"'
        )

    def _handle_live_event(self, event: dict[str, object]) -> None:
        event_type = str(event.get('type', 'event'))
        if event_type == 'message_start':
            self._update_activity('model stream connected')
            return
        if event_type == 'content_delta':
            self.live_assistant_text += str(event.get('delta') or '')
            body = _normalize_for_display(self.live_assistant_text)
            visible_body = body or 'streaming...'
            if self.live_assistant_block is None:
                self.live_assistant_block = self._append_entry('assistant', visible_body)
            else:
                self.live_assistant_block.update_body(visible_body)
                self._replace_transcript_entry(self.live_assistant_block, visible_body)
            self._update_activity(f'streaming assistant response... {len(self.live_assistant_text)} chars')
            return
        if event_type == 'tool_start':
            name = str(event.get('tool_name') or 'tool')
            self.live_tool_counts[name] = self.live_tool_counts.get(name, 0) + 1
            self._update_activity(f'tool running: {name}')
            return
        if event_type == 'tool_delta':
            name = str(event.get('tool_name') or 'tool')
            self._update_activity(f'tool output: {name}')
            return
        if event_type == 'tool_result':
            name = str(event.get('tool_name') or 'tool')
            ok = event.get('ok')
            status = 'ok' if ok is True else 'failed' if ok is False else 'done'
            self._update_activity(f'tool finished: {name} {status}')
            return
        if event_type in {'compact_boundary', 'reactive_compact_boundary'}:
            count = event.get('compacted_message_count') or 'some'
            self._update_activity(f'context compacted: {count} messages')
            return
        if event_type == 'snip_boundary':
            count = event.get('snipped_message_count') or 'some'
            self._update_activity(f'context snipped: {count} messages')
            return
        if event_type == 'tool_loop_guard':
            self._update_activity('tool loop guard stopped repeated no-op tool calls')
            self._append_entry('system', 'Tool loop guard stopped repeated no-op tool calls.')


    def _update_activity(self, text: str) -> None:
        try:
            self.query_one('#activity-bar', Static).update(text)
        except Exception:
            pass

    def _refresh_running_status(self) -> None:
        if self.busy:
            self._update_status('running')
        elif self.voice_recording is not None:
            self._update_status('listening')

    def _update_status(self, state: str) -> None:
        elapsed_seconds = None
        if self.busy_started_at is not None:
            elapsed_seconds = int(max(time.monotonic() - self.busy_started_at, 0))
        elif self.voice_started_at is not None:
            elapsed_seconds = int(max(time.monotonic() - self.voice_started_at, 0))
        text = _build_status_text(
            state=state,
            model_config=self.controller.model_config,
            runtime_config=self.controller.runtime_config,
            session_id=self.controller.session_id,
            elapsed_seconds=elapsed_seconds,
        )
        self.query_one('#status-bar', Static).update(text)

    def action_new_session(self) -> None:
        self.controller.start_new_session()
        self._append_system('started new session')
        self._update_status('ready')

    def action_show_status(self) -> None:
        self._append_system('\n'.join(self.controller.status_lines()))

    def action_prompt_history_previous(self) -> None:
        if not self.prompt_history:
            return
        prompt = self.query_one('#prompt', PromptTextArea)
        if self.prompt_history_index is None:
            self.prompt_history_draft = prompt.value
            self.prompt_history_index = len(self.prompt_history) - 1
        else:
            self.prompt_history_index = max(self.prompt_history_index - 1, 0)
        self._set_prompt_value(self.prompt_history[self.prompt_history_index])

    def action_prompt_history_next(self) -> None:
        if self.prompt_history_index is None:
            return
        next_index = self.prompt_history_index + 1
        if next_index >= len(self.prompt_history):
            self.prompt_history_index = None
            self._set_prompt_value(self.prompt_history_draft)
            self.prompt_history_draft = ''
            return
        self.prompt_history_index = next_index
        self._set_prompt_value(self.prompt_history[self.prompt_history_index])

    def action_focus_find(self) -> None:
        prompt = self.query_one('#prompt', PromptTextArea)
        if prompt.value and not prompt.value.startswith(('/find ', ':find ')):
            self._append_system('Finish or clear the current prompt before starting transcript search.')
            prompt.focus()
            return
        self._set_prompt_value('/find ')

    def action_copy_last_turn(self) -> None:
        self._copy_text(self._format_transcript_entries(self._last_conversation_turn()), 'Last turn')

    def action_export_transcript(self) -> None:
        export_dir = self.controller.runtime_config.session_directory / 'exports'
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        path = export_dir / f'clawed-transcript-{timestamp}.md'
        path.write_text(self._format_transcript_markdown(), encoding='utf-8')
        self._append_system(f'Transcript exported: {path}')

    def action_toggle_write(self) -> None:
        self.controller.toggle_write()
        self._append_system(f'write={self.controller.runtime_config.permissions.allow_file_write}')
        self._update_status('ready')

    def _request_or_toggle_shell(self) -> None:
        if self.controller.runtime_config.permissions.allow_shell_commands:
            self.controller.toggle_shell()
            if self.pending_permission_confirmation == 'shell':
                self.pending_permission_confirmation = None
            self._append_system('shell=False')
            self._update_status('ready')
            return
        self.pending_permission_confirmation = 'shell'
        self._append_system('Confirm shell enablement: type /confirm shell to allow bash tool calls for this TUI session.')
        self._update_status('ready')

    def _request_or_toggle_unsafe(self) -> None:
        if self.controller.runtime_config.permissions.allow_destructive_shell_commands:
            self.controller.toggle_unsafe()
            if self.pending_permission_confirmation == 'unsafe':
                self.pending_permission_confirmation = None
            self._append_system('unsafe=False')
            self._update_status('ready')
            return
        self.pending_permission_confirmation = 'unsafe'
        self._append_system('Confirm unsafe mode: type /confirm unsafe to allow destructive shell commands for this TUI session.')
        self._update_status('ready')

    def _confirm_permission(self, permission: str) -> None:
        if self.pending_permission_confirmation != permission:
            self._append_system(f'No pending {permission} confirmation.')
            return
        self.pending_permission_confirmation = None
        if permission == 'shell':
            if not self.controller.runtime_config.permissions.allow_shell_commands:
                self.controller.toggle_shell()
            self._append_system(f'shell={self.controller.runtime_config.permissions.allow_shell_commands}')
        elif permission == 'unsafe':
            if not self.controller.runtime_config.permissions.allow_destructive_shell_commands:
                self.controller.toggle_unsafe()
            self._append_system(f'unsafe={self.controller.runtime_config.permissions.allow_destructive_shell_commands}')
        self._update_status('ready')

    def action_toggle_shell(self) -> None:
        self._request_or_toggle_shell()

    def action_toggle_unsafe(self) -> None:
        self._request_or_toggle_unsafe()

    async def action_voice_input(self) -> None:
        if self.busy:
            return
        prompt_input = self.query_one('#prompt', PromptTextArea)
        if self.voice_recording is None:
            try:
                self.voice_recording = await asyncio.to_thread(self.controller.start_voice_recording)
            except Exception as exc:
                self._append_system(f'voice error: {exc}')
                self._update_status('ready')
                return
            self.voice_started_at = time.monotonic()
            self._append_system('Voice recording started. Press F4 again to stop and transcribe.')
            self._update_status('listening')
            prompt_input.focus()
            return

        recording = self.voice_recording
        self.voice_recording = None
        elapsed = int(max(time.monotonic() - (self.voice_started_at or time.monotonic()), 0))
        self.voice_started_at = None
        prompt_input.disabled = True
        self._append_system(f'Voice recording stopped after {elapsed}s. Transcribing...')
        self._update_status('transcribing')
        try:
            text = await asyncio.to_thread(recording.stop_and_transcribe)
        except Exception as exc:
            self._append_system(f'voice error: {exc}')
        else:
            existing = prompt_input.value.strip()
            prompt_input.value = f'{existing} {text}'.strip() if existing else text
            self._append_system('Voice input inserted into prompt. Press Enter to send.')
        finally:
            prompt_input.disabled = False
            prompt_input.focus()
            self._update_status('ready')

    def action_page_up(self) -> None:
        self.query_one('#transcript', VerticalScroll).scroll_page_up(animate=False)

    def action_page_down(self) -> None:
        self.query_one('#transcript', VerticalScroll).scroll_page_down(animate=False)

    def action_scroll_home(self) -> None:
        self.query_one('#transcript', VerticalScroll).scroll_home(animate=False)

    def action_scroll_end(self) -> None:
        self.query_one('#transcript', VerticalScroll).scroll_end(animate=False)

    def _splash_text(self) -> str:
        return f'{DRAGON_ART}\n\nDRAGON CLAWED TUI  |  local or remote coding agent  |  Textual interface'



def _build_status_text(
    *,
    state: str,
    model_config: ModelConfig,
    runtime_config: AgentRuntimeConfig,
    session_id: str | None,
    elapsed_seconds: int | None = None,
) -> str:
    permissions = runtime_config.permissions
    state_text = state
    if state == 'running' and elapsed_seconds is not None:
        state_text = f'running {elapsed_seconds}s'
    return (
        f'{state_text} | model={model_config.model} '
        f'| timeout={model_config.timeout_seconds:g}s '
        f'| session={session_id or "<new>"} '
        f'| cwd={runtime_config.cwd} '
        f'| write={permissions.allow_file_write} '
        f'shell={permissions.allow_shell_commands} '
        f'unsafe={permissions.allow_destructive_shell_commands}'
    )


def _summarize_run_events(events: tuple[dict[str, object], ...]) -> tuple[str, str]:
    tool_results: list[str] = []
    tool_starts: dict[str, str] = {}
    delegates: list[str] = []
    for event in events:
        event_type = str(event.get('type', 'event'))
        if event_type == 'tool_start':
            call_id = str(event.get('tool_call_id') or len(tool_starts))
            tool_starts[call_id] = str(event.get('tool_name') or 'tool')
        elif event_type == 'tool_result':
            name = str(event.get('tool_name') or 'tool')
            ok = event.get('ok')
            status = 'ok' if ok is True else 'failed' if ok is False else 'done'
            tool_results.append(f'{name} {status}')
        elif event_type == 'delegate_subtask_result':
            label = event.get('label') or event.get('index') or 'subtask'
            stop = event.get('stop_reason') or 'done'
            delegates.append(f'{label} {stop}')
        elif event_type == 'delegate_group_result':
            group_id = event.get('group_id') or 'group'
            completed = event.get('completed_children')
            failed = event.get('failed_children')
            delegates.append(f'{group_id}: {completed} completed, {failed} failed')
    if not tool_results and tool_starts:
        tool_results.append(f'{len(tool_starts)} started')
    return _compact_summary(tool_results), _compact_summary(delegates)


def _summarize_control_events(events: tuple[dict[str, object], ...], *, stop_reason: str | None = None) -> str:
    items: list[str] = []
    for event in events:
        event_type = str(event.get('type', 'event'))
        if event_type in {'compact_boundary', 'reactive_compact_boundary'}:
            count = event.get('compacted_message_count')
            removed = event.get('estimated_tokens_removed')
            label = f'compacted {count} messages' if count is not None else 'compacted history'
            if removed is not None:
                label = f'{label} (removed ~{removed} tokens)'
            items.append(label)
        elif event_type in {'snip_boundary', 'reactive_snip_boundary'}:
            count = event.get('snipped_message_count')
            items.append(f'snipped {count} messages' if count is not None else 'snipped history')
        elif event_type == 'reactive_compact_retry':
            turn = event.get('turn_index')
            items.append(f'reactive compact retry on turn {turn}' if turn is not None else 'reactive compact retry')
        elif event_type == 'tool_result' and event.get('ok') is False:
            items.append(f'{event.get("tool_name") or "tool"} failed')
        elif event_type == 'tool_loop_guard':
            name = event.get('tool_name') or 'tool'
            repeat = event.get('repeat_count')
            items.append(f'tool loop guard: {name} repeated {repeat}x' if repeat is not None else f'tool loop guard: {name}')
        elif event_type == 'empty_response':
            turn = event.get('turn_index')
            items.append(f'empty response on turn {turn}' if turn is not None else 'empty response')
        elif event_type in {'tool_budget_exceeded', 'task_budget_exceeded'}:
            name = event.get('tool_name') or 'tool'
            reason = event.get('reason') or 'budget exceeded'
            items.append(f'{name} blocked: {reason}')
        elif event_type == 'model_error':
            items.append(str(event.get('message') or 'model error'))
    if stop_reason == 'backend_error':
        items.append('backend error')
    elif stop_reason == 'budget_exceeded':
        items.append('runtime budget exceeded')
    return _compact_summary(items, limit=8)


def _compact_summary(items: list[str], *, limit: int = 5) -> str:
    if not items:
        return ''
    shown = items[:limit]
    suffix = f', +{len(items) - limit} more' if len(items) > limit else ''
    return ', '.join(shown) + suffix

def _normalize_for_display(text: str) -> str:
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


def run_textual_tui(controller: TuiController) -> int:
    app = ClawedTextualApp(controller)
    app.run()
    return 0
