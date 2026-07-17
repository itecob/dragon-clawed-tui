from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.agent_types import AgentRunResult, AgentRuntimeConfig, ModelConfig
from src.session_store import StoredAgentSession, save_agent_session
from src.textual_tui_app import ClawedTextualApp, MessageBlock, PromptTextArea, _build_status_text, _normalize_for_display, _prompt_content_height, _summarize_control_events, _summarize_run_events, _theme_css
from src.tui_app import TuiController


class FakeController(TuiController):
    def __init__(self, workspace: Path) -> None:
        super().__init__(
            model_config=ModelConfig(model='fake-model'),
            runtime_config=AgentRuntimeConfig(cwd=workspace),
        )
        self.prompts: list[str] = []

    def run_prompt(self, prompt: str, *, event_callback=None) -> AgentRunResult:
        self.prompts.append(prompt)
        if event_callback is not None:
            event_callback({'type': 'content_delta', 'delta': 'Live '})
            event_callback({'type': 'content_delta', 'delta': 'Done.'})
        return AgentRunResult(final_output='assistant\nLive Done.', turns=1, tool_calls=0, transcript=(), session_id='fake-session')


class FakeRecording:
    def __init__(self) -> None:
        self.stopped = False

    def stop_and_transcribe(self) -> str:
        self.stopped = True
        return 'voice prompt text'


class FakeVoiceService:
    def __init__(self) -> None:
        self.recordings: list[FakeRecording] = []

    def start_recording(self) -> FakeRecording:
        recording = FakeRecording()
        self.recordings.append(recording)
        return recording

    def capture_and_transcribe(self) -> str:
        return 'voice prompt text'


class RaisingController(TuiController):
    def __init__(self, workspace: Path) -> None:
        super().__init__(
            model_config=ModelConfig(model='fake-model'),
            runtime_config=AgentRuntimeConfig(cwd=workspace),
        )

    def run_prompt(self, prompt: str, *, event_callback=None) -> AgentRunResult:
        raise RuntimeError('backend unavailable')


class TextualTuiAppTests(unittest.IsolatedAsyncioTestCase):
    async def test_textual_app_mounts_status_and_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                self.assertIn('fake-model', app.query_one('#status-bar').renderable)
                self.assertEqual(app.query_one('#prompt-label').renderable, 'USER PROMPT')
                prompt = app.query_one('#prompt', PromptTextArea)
                self.assertTrue(prompt.soft_wrap)
                self.assertEqual(prompt.show_line_numbers, False)


    def test_textual_theme_css_uses_cohesive_default_and_named_variants(self) -> None:
        default_css = _theme_css('codex')
        light_css = _theme_css('light')
        unknown_css = _theme_css('unknown-theme')

        self.assertIn('#prompt-row', default_css)
        self.assertIn('border-top: solid', default_css)
        self.assertIn('.assistant .role-label', default_css)
        self.assertIn('#f6f1e8', light_css)
        self.assertEqual(default_css, unknown_css)

    def test_prompt_content_height_starts_compact_and_grows_to_cap(self) -> None:
        self.assertEqual(_prompt_content_height('', width=80), 1)
        self.assertEqual(_prompt_content_height('one line', width=80), 1)
        self.assertEqual(_prompt_content_height('one\ntwo\nthree', width=80), 3)
        self.assertEqual(_prompt_content_height('x' * 500, width=40), 6)

    async def test_textual_app_prompt_grows_when_text_area_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = FakeController(workspace)
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                self.assertEqual(prompt.styles.height.value, 1)

                prompt.insert('one\ntwo\nthree\nfour')
                await pilot.pause()

                self.assertEqual(prompt.styles.height.value, 4)

    async def test_textual_app_multiline_prompt_preserves_wrapped_and_explicit_newlines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = FakeController(workspace)
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = 'first line\nsecond line with enough words to wrap in a narrow prompt area'
                await pilot.press('enter')
                await pilot.pause()
                await pilot.pause()

        self.assertEqual(controller.prompts, ['first line\nsecond line with enough words to wrap in a narrow prompt area'])


    async def test_textual_app_voice_action_appends_to_multiline_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
                voice_input_service=FakeVoiceService(),
            )
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = 'existing line'
                await pilot.press('f4')
                await pilot.pause()
                await pilot.press('f4')
                await pilot.pause()
                await pilot.pause()

        self.assertEqual(prompt.value, 'existing line voice prompt text')


    async def test_textual_app_shows_immediate_model_activity_before_stream_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = FakeController(workspace)
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = 'hello'
                await pilot.press('enter')
                await pilot.pause()
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

        self.assertTrue(any('Model request started' in body for body in bodies))

    async def test_textual_app_shows_placeholder_for_hidden_initial_stream_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app._handle_live_event({'type': 'content_delta', 'delta': 'assistant'})
                await pilot.pause()
                assistant_blocks = [
                    block for block in app.query(MessageBlock) if block.entry.role == 'assistant'
                ]

        self.assertEqual(len(assistant_blocks), 1)
        self.assertIn('streaming', assistant_blocks[0].entry.body)


    async def test_textual_app_renders_live_stream_without_duplicate_final_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = FakeController(workspace)
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.query_one('#prompt', PromptTextArea).value = 'hello'
                await pilot.press('enter')
                await pilot.pause()
                await pilot.pause()
                roles = [str(widget.renderable) for widget in app.query('.role-label')]
                assistant_blocks = [
                    block for block in app.query(MessageBlock) if block.entry.role == 'assistant'
                ]

        self.assertEqual(roles.count('ASSISTANT'), 1)
        self.assertEqual(assistant_blocks[0].entry.body, 'Live Done.')


    async def test_textual_app_lists_recent_sessions_from_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            session_dir = workspace / '.port_sessions' / 'agent'
            save_agent_session(
                StoredAgentSession(
                    session_id='session-alpha',
                    model_config={'model': 'fake-model'},
                    runtime_config={'cwd': str(workspace)},
                    system_prompt_parts=(),
                    user_context={},
                    system_context={},
                    messages=({'role': 'user', 'content': 'Review the TUI'},),
                    turns=3,
                    tool_calls=2,
                    usage={'total_tokens': 42},
                    total_cost_usd=0.0,
                    file_history=(),
                ),
                directory=session_dir,
            )
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace, session_directory=session_dir),
            )
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = '/sessions'
                await pilot.press('enter')
                await pilot.pause()
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

        self.assertTrue(any('1. session-alpha' in body and 'Review the TUI' in body for body in bodies))

    async def test_textual_app_filters_sessions_and_resumes_filtered_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            session_dir = workspace / '.port_sessions' / 'agent'
            for session_id, prompt_text in (('alpha-session', 'Alpha planning'), ('beta-session', 'Beta repair')):
                save_agent_session(
                    StoredAgentSession(
                        session_id=session_id,
                        model_config={'model': 'fake-model'},
                        runtime_config={'cwd': str(workspace)},
                        system_prompt_parts=(),
                        user_context={},
                        system_context={},
                        messages=({'role': 'user', 'content': prompt_text},),
                        turns=1,
                        tool_calls=0,
                        usage={},
                        total_cost_usd=0.0,
                        file_history=(),
                    ),
                    directory=session_dir,
                )
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace, session_directory=session_dir),
            )
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = '/sessions beta'
                await pilot.press('enter')
                await pilot.pause()
                picker_bodies = [str(widget.renderable) for widget in app.query('.message-body')]
                prompt.value = '/resume 1'
                await pilot.press('enter')
                await pilot.pause()

        self.assertEqual(controller.session_id, 'beta-session')
        joined = '\n'.join(picker_bodies)
        self.assertIn('Filter: beta', joined)
        self.assertIn('1. beta-session', joined)
        self.assertNotIn('alpha-session', joined)

    async def test_textual_app_resumes_recent_session_by_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            session_dir = workspace / '.port_sessions' / 'agent'
            save_agent_session(
                StoredAgentSession(
                    session_id='session-beta',
                    model_config={'model': 'fake-model'},
                    runtime_config={'cwd': str(workspace)},
                    system_prompt_parts=(),
                    user_context={},
                    system_context={},
                    messages=({'role': 'user', 'content': 'Resume me'},),
                    turns=1,
                    tool_calls=0,
                    usage={},
                    total_cost_usd=0.0,
                    file_history=(),
                ),
                directory=session_dir,
            )
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace, session_directory=session_dir),
            )
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = '/resume 1'
                await pilot.press('enter')
                await pilot.pause()

        self.assertEqual(controller.session_id, 'session-beta')

    async def test_textual_app_prompt_history_restores_previous_and_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = FakeController(workspace)
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = 'first prompt'
                await pilot.press('enter')
                await pilot.pause()
                await pilot.pause()
                prompt.value = 'second prompt'
                await pilot.press('enter')
                await pilot.pause()
                await pilot.pause()
                prompt.value = 'draft prompt'

                app.action_prompt_history_previous()
                self.assertEqual(prompt.value, 'second prompt')
                app.action_prompt_history_previous()
                self.assertEqual(prompt.value, 'first prompt')
                app.action_prompt_history_next()
                self.assertEqual(prompt.value, 'second prompt')
                app.action_prompt_history_next()
                self.assertEqual(prompt.value, 'draft prompt')

        self.assertEqual(controller.prompts, ['first prompt', 'second prompt'])

    async def test_textual_app_prompt_history_ignores_local_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = FakeController(workspace)
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = '/tools'
                await pilot.press('enter')
                await pilot.pause()
                prompt.value = '/find tools'
                await pilot.press('enter')
                await pilot.pause()
                prompt.value = 'real prompt'
                await pilot.press('enter')
                await pilot.pause()
                await pilot.pause()

        self.assertEqual(app.prompt_history, ['real prompt'])

    async def test_textual_app_prompt_history_ignores_duplicate_consecutive_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = FakeController(workspace)
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = 'repeat prompt'
                await pilot.press('enter')
                await pilot.pause()
                await pilot.pause()
                prompt.value = 'repeat prompt'
                await pilot.press('enter')
                await pilot.pause()
                await pilot.pause()

        self.assertEqual(app.prompt_history, ['repeat prompt'])

    async def test_textual_app_requires_confirmation_before_enabling_shell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = FakeController(workspace)
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = '/shell'
                await pilot.press('enter')
                await pilot.pause()
                self.assertFalse(controller.runtime_config.permissions.allow_shell_commands)
                prompt.value = '/confirm shell'
                await pilot.press('enter')
                await pilot.pause()
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

        self.assertTrue(controller.runtime_config.permissions.allow_shell_commands)
        self.assertTrue(any('Confirm shell enablement' in body for body in bodies))
        self.assertTrue(any('shell=True' in body for body in bodies))

    async def test_textual_app_requires_confirmation_before_enabling_unsafe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = FakeController(workspace)
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = '/unsafe'
                await pilot.press('enter')
                await pilot.pause()
                self.assertFalse(controller.runtime_config.permissions.allow_destructive_shell_commands)
                prompt.value = '/confirm unsafe'
                await pilot.press('enter')
                await pilot.pause()
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

        self.assertTrue(controller.runtime_config.permissions.allow_destructive_shell_commands)
        self.assertTrue(any('Confirm unsafe mode' in body for body in bodies))
        self.assertTrue(any('unsafe=True' in body for body in bodies))

    async def test_textual_app_disables_shell_without_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = FakeController(workspace)
            controller.toggle_shell()
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = '/shell'
                await pilot.press('enter')
                await pilot.pause()

        self.assertFalse(controller.runtime_config.permissions.allow_shell_commands)


    async def test_textual_app_handles_agents_command_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='main-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace, delegate_model='helper-model'),
            )
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = '/agents'
                await pilot.press('enter')
                await pilot.pause()
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

        self.assertTrue(any('# Agents' in body and 'Delegate model: helper-model' in body for body in bodies))

    async def test_textual_app_handles_diff_command_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = FakeController(workspace)
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = '/diff'
                await pilot.press('enter')
                await pilot.pause()
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

        self.assertEqual(controller.prompts, [])
        self.assertTrue(any('# Diff' in body for body in bodies))

    async def test_textual_app_handles_agent_report_commands_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = FakeController(workspace)
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = '/tools'
                await pilot.press('enter')
                await pilot.pause()
                prompt.value = '/permissions'
                await pilot.press('enter')
                await pilot.pause()
                prompt.value = '/context'
                await pilot.press('enter')
                await pilot.pause()
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

        self.assertEqual(controller.prompts, [])
        self.assertTrue(any('# Tools' in body and '`web_search`' in body for body in bodies))
        self.assertTrue(any('# Permissions' in body and 'Shell commands' in body for body in bodies))
        self.assertTrue(any('# Context' in body for body in bodies))

    async def test_textual_app_help_lists_local_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = FakeController(workspace)
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = '/help'
                await pilot.press('enter')
                await pilot.pause()
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

        self.assertEqual(controller.prompts, [])
        help_text = '\n'.join(bodies)
        self.assertIn('/tools', help_text)
        self.assertIn('/permissions', help_text)
        self.assertIn('/context', help_text)
        self.assertIn('/sessions', help_text)
        self.assertIn('/export', help_text)


    async def test_textual_app_finds_transcript_matches_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = FakeController(workspace)
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app._append_entry('user', 'Alpha appears in the first prompt')
                app._append_entry('assistant', 'No match here')
                app._append_entry('assistant', 'Second alpha result is here')
                prompt = app.query_one('#prompt', PromptTextArea)

                prompt.value = '/find alpha'
                await pilot.press('enter')
                await pilot.pause()
                prompt.value = '/find alpha'
                await pilot.press('enter')
                await pilot.pause()
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

        self.assertEqual(controller.prompts, [])
        self.assertTrue(any('Search 1/2: USER contains "alpha"' in body for body in bodies))
        self.assertTrue(any('Search 2/2: ASSISTANT contains "alpha"' in body for body in bodies))

    async def test_textual_app_reports_find_usage_and_misses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = FakeController(workspace)
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app._append_entry('assistant', 'Only local transcript text')
                prompt = app.query_one('#prompt', PromptTextArea)

                prompt.value = '/find'
                await pilot.press('enter')
                await pilot.pause()
                prompt.value = '/find absent'
                await pilot.press('enter')
                await pilot.pause()
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

        self.assertEqual(controller.prompts, [])
        self.assertTrue(any('Usage: /find <text>' in body for body in bodies))
        self.assertTrue(any('No transcript matches for "absent"' in body for body in bodies))

    async def test_textual_app_ctrl_f_seeds_find_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = FakeController(workspace)
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press('ctrl+f')
                prompt = app.query_one('#prompt', PromptTextArea)

        self.assertEqual(prompt.value, '/find ')


    async def test_textual_app_submits_prompt_to_controller(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = FakeController(workspace)
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.query_one('#prompt', PromptTextArea).value = 'hello'
                await pilot.press('enter')
                await pilot.pause()
                await pilot.pause()

        self.assertEqual(controller.prompts, ['hello'])


    async def test_textual_app_skips_empty_assistant_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app._append_result(AgentRunResult(final_output='assistant\n<think>\n\n</think>\n', turns=1, tool_calls=0, transcript=()))
                await pilot.pause()
                roles = [str(widget.renderable) for widget in app.query('.role-label')]

        self.assertNotIn('ASSISTANT', roles)
        self.assertIn('SYSTEM', roles)


    async def test_textual_app_voice_action_inserts_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
                voice_input_service=FakeVoiceService(),
            )
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press('f4')
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                self.assertEqual(prompt.value, '')
                await pilot.press('f4')
                await pilot.pause()
                await pilot.pause()

        self.assertEqual(prompt.value, 'voice prompt text')
        self.assertFalse(prompt.disabled)


    async def test_textual_app_reenables_prompt_after_controller_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = RaisingController(workspace)
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = 'hello'
                await pilot.press('enter')
                await pilot.pause()
                await pilot.pause()
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

        self.assertFalse(prompt.disabled)
        self.assertTrue(any('error: backend unavailable' in body for body in bodies))


    def test_summarize_control_events_reports_runtime_budget_event_name(self) -> None:
        summary = _summarize_control_events((
            {'type': 'task_budget_exceeded', 'tool_name': 'bash', 'reason': 'max tool calls reached'},
        ))

        self.assertIn('bash blocked: max tool calls reached', summary)

    def test_summarize_control_events_reports_context_and_failures(self) -> None:
        summary = _summarize_control_events((
            {'type': 'compact_boundary', 'compacted_message_count': 4, 'estimated_tokens_removed': 1200},
            {'type': 'snip_boundary', 'snipped_message_count': 2},
            {'type': 'reactive_compact_retry', 'turn_index': 3},
            {'type': 'tool_result', 'tool_name': 'bash', 'ok': False},
            {'type': 'tool_loop_guard', 'tool_name': 'read_file', 'repeat_count': 3},
            {'type': 'empty_response', 'turn_index': 4},
        ))

        self.assertIn('compacted 4 messages', summary)
        self.assertIn('removed ~1200 tokens', summary)
        self.assertIn('snipped 2 messages', summary)
        self.assertIn('reactive compact retry on turn 3', summary)
        self.assertIn('bash failed', summary)
        self.assertIn('tool loop guard: read_file repeated 3x', summary)
        self.assertIn('empty response on turn 4', summary)

    async def test_textual_app_renders_backend_error_control_summary_from_stop_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app._append_result(AgentRunResult(
                    final_output='Backend error: timeout',
                    turns=1,
                    tool_calls=0,
                    transcript=(),
                    stop_reason='backend_error',
                    events=(),
                ))
                await pilot.pause()
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

        self.assertTrue(any('Control:' in body and 'backend error' in body for body in bodies))

    async def test_textual_app_renders_control_summary_for_context_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app._append_result(AgentRunResult(
                    final_output='Done.',
                    turns=3,
                    tool_calls=2,
                    transcript=(),
                    stop_reason='stop',
                    events=(
                        {'type': 'compact_boundary', 'compacted_message_count': 4, 'estimated_tokens_removed': 1200},
                        {'type': 'tool_result', 'tool_name': 'bash', 'ok': False},
                        {'type': 'tool_loop_guard', 'tool_name': 'bash', 'repeat_count': 3},
                    ),
                ))
                await pilot.pause()
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

        self.assertTrue(any('Control:' in body and 'compacted 4 messages' in body and 'bash failed' in body for body in bodies))


    def test_summarize_run_events_compacts_tool_events(self) -> None:
        tool_summary, delegate_summary = _summarize_run_events((
            {'type': 'tool_start', 'tool_name': 'list_dir', 'tool_call_id': 'a'},
            {'type': 'tool_delta', 'tool_name': 'list_dir', 'delta': 'lots of noisy output'},
            {'type': 'tool_result', 'tool_name': 'list_dir', 'ok': True},
            {'type': 'tool_result', 'tool_name': 'read_file', 'ok': False},
            {'type': 'delegate_subtask_result', 'label': 'scan', 'stop_reason': 'stop'},
        ))

        self.assertEqual(tool_summary, 'list_dir ok, read_file failed')
        self.assertEqual(delegate_summary, 'scan stop')

    async def test_textual_app_renders_tool_summary_not_tool_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app._append_result(AgentRunResult(
                    final_output='Done.',
                    turns=1,
                    tool_calls=1,
                    transcript=(),
                    events=(
                        {'type': 'tool_start', 'tool_name': 'list_dir', 'tool_call_id': 'a'},
                        {'type': 'tool_result', 'tool_name': 'list_dir', 'ok': True},
                    ),
                ))
                await pilot.pause()
                roles = [str(widget.renderable) for widget in app.query('.role-label')]
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

        self.assertNotIn('TOOL', roles)
        self.assertTrue(any('Tool activity: list_dir ok' in body for body in bodies))

    async def test_textual_app_reports_incomplete_response_after_tool_activity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app._append_result(AgentRunResult(
                    final_output='',
                    turns=4,
                    tool_calls=10,
                    transcript=(),
                    events=(
                        {'type': 'tool_result', 'tool_name': 'read_file', 'ok': True},
                    ),
                ))
                await pilot.pause()
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

        self.assertTrue(any('No final assistant response was produced after tool activity.' in body for body in bodies))


    async def test_textual_app_routes_live_tool_updates_to_activity_bar_not_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app._handle_live_event({'type': 'tool_start', 'tool_name': 'read_file'})
                await pilot.pause()
                activity = str(app.query_one('#activity-bar').renderable)
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

        self.assertIn('tool running: read_file', activity)
        self.assertFalse(any('Tool started: read_file' in body for body in bodies))

    async def test_textual_app_activity_bar_shows_stream_connection_before_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app._handle_live_event({'type': 'message_start'})
                await pilot.pause()
                activity = str(app.query_one('#activity-bar').renderable)

        self.assertIn('model stream connected', activity)

    async def test_textual_app_copies_last_turn_to_clipboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )
            app = ClawedTextualApp(controller)
            copied: list[str] = []
            app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
            async with app.run_test() as pilot:
                await pilot.pause()
                app._append_entry('user', 'Inspect the project')
                app._append_entry('assistant', 'The project uses Textual.')
                app.action_copy_last_turn()

        self.assertEqual(copied, ['USER: Inspect the project\n\nASSISTANT: The project uses Textual.'])

    async def test_textual_app_writes_copy_fallback_when_clipboard_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            session_dir = workspace / 'sessions'
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace, session_directory=session_dir),
            )
            app = ClawedTextualApp(controller)

            def raise_clipboard(_text: str) -> None:
                raise RuntimeError('clipboard unavailable')

            app.copy_to_clipboard = raise_clipboard  # type: ignore[method-assign]
            async with app.run_test() as pilot:
                await pilot.pause()
                app._append_entry('user', 'Copy this prompt')
                app._append_entry('assistant', 'Copy this response')
                app.action_copy_last_turn()
                await pilot.pause()
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

            fallbacks = sorted((session_dir / 'exports').glob('clawed-last-turn-*.txt'))
            self.assertEqual(len(fallbacks), 1)
            self.assertIn('USER: Copy this prompt', fallbacks[0].read_text(encoding='utf-8'))
            self.assertIn('ASSISTANT: Copy this response', fallbacks[0].read_text(encoding='utf-8'))
            self.assertTrue(any('clipboard unavailable' in body and 'written:' in body for body in bodies))

    async def test_textual_app_new_session_clears_transcript_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = FakeController(workspace)
            app = ClawedTextualApp(controller)
            copied: list[str] = []
            app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
            async with app.run_test() as pilot:
                await pilot.pause()
                app._append_entry('user', 'Old session prompt')
                app._append_entry('assistant', 'Old session answer')
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = '/new'
                await pilot.press('enter')
                await pilot.pause()
                app.action_copy_last_turn()
                app._find_transcript('Old session')
                await pilot.pause()
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

        self.assertEqual(copied, [])
        self.assertEqual([entry.body for entry in app.transcript_entries if entry.role in {'user', 'assistant'}], [])
        self.assertTrue(any('Last turn: nothing to copy' in body for body in bodies))
        self.assertTrue(any('No transcript matches for "Old session"' in body for body in bodies))

    async def test_textual_app_resume_session_clears_transcript_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            session_dir = workspace / '.port_sessions' / 'agent'
            save_agent_session(
                StoredAgentSession(
                    session_id='session-gamma',
                    model_config={'model': 'fake-model'},
                    runtime_config={'cwd': str(workspace)},
                    system_prompt_parts=(),
                    user_context={},
                    system_context={},
                    messages=({'role': 'user', 'content': 'Resume target'},),
                    turns=1,
                    tool_calls=0,
                    usage={},
                    total_cost_usd=0.0,
                    file_history=(),
                ),
                directory=session_dir,
            )
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace, session_directory=session_dir),
            )
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app._append_entry('user', 'Previous visible prompt')
                prompt = app.query_one('#prompt', PromptTextArea)
                prompt.value = '/resume 1'
                await pilot.press('enter')
                await pilot.pause()
                app._find_transcript('Previous visible prompt')
                await pilot.pause()
                bodies = [str(widget.renderable) for widget in app.query('.message-body')]

        self.assertEqual(controller.session_id, 'session-gamma')
        self.assertEqual([entry.body for entry in app.transcript_entries if entry.role == 'user'], [])
        self.assertTrue(any('No transcript matches for "Previous visible prompt"' in body for body in bodies))

    async def test_textual_app_exports_unique_paths_within_same_second(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            session_dir = workspace / 'sessions'
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace, session_directory=session_dir),
            )
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app._append_entry('user', 'Hello')
                app.action_export_transcript()
                app.action_export_transcript()

            exports = sorted((session_dir / 'exports').glob('clawed-transcript-*.md'))
            self.assertEqual(len(exports), 2)

    async def test_textual_app_exports_transcript_to_session_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            session_dir = workspace / 'sessions'
            controller = TuiController(
                model_config=ModelConfig(model='fake-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace, session_directory=session_dir),
            )
            app = ClawedTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app._append_entry('user', 'Hello')
                app._append_entry('assistant', 'Hi')
                app.action_export_transcript()

            exports = sorted((session_dir / 'exports').glob('clawed-transcript-*.md'))
            self.assertEqual(len(exports), 1)
            self.assertIn('## USER\n\nHello', exports[0].read_text(encoding='utf-8'))
            self.assertIn('## ASSISTANT\n\nHi', exports[0].read_text(encoding='utf-8'))

    def test_build_status_text_includes_running_elapsed_and_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_config = AgentRuntimeConfig(cwd=Path(tmp_dir))
            text = _build_status_text(
                state='running',
                model_config=ModelConfig(model='fake-model', timeout_seconds=60),
                runtime_config=runtime_config,
                session_id='abc123',
                elapsed_seconds=12,
            )

        self.assertIn('running 12s', text)
        self.assertIn('timeout=60s', text)
        self.assertIn('session=abc123', text)


    def test_normalize_for_display_removes_empty_think_and_role(self) -> None:
        self.assertEqual(
            _normalize_for_display('assistant\n<think>\n\n</think>\n\nHello'),
            'Hello',
        )


if __name__ == '__main__':
    unittest.main()
