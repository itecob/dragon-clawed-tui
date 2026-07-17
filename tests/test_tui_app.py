from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from src.agent_types import AgentRunResult, AgentRuntimeConfig, ModelConfig
from src.session_store import StoredAgentSession, save_agent_session
from src.tui_app import TuiController, _CursesTui, _format_result_for_display, _handle_plain_command, _handle_rich_command, _normalize_assistant_text, _wrap_display_line, render_plain_event


class FakeAgent:
    prompts: list[tuple[str, str]]

    def __init__(self, runtime_config: AgentRuntimeConfig) -> None:
        self.runtime_config = runtime_config
        self.prompts = []

    def run(self, prompt: str, *, event_callback=None) -> AgentRunResult:
        self.prompts.append(('run', prompt))
        session_id = 'session-one'
        save_agent_session(
            StoredAgentSession(
                session_id=session_id,
                model_config={'model': 'fake'},
                runtime_config={'cwd': str(self.runtime_config.cwd)},
                system_prompt_parts=(),
                user_context={},
                system_context={},
                messages=(),
                turns=1,
                tool_calls=0,
                usage={},
                total_cost_usd=0.0,
                file_history=(),
            ),
            directory=self.runtime_config.session_directory,
        )
        return AgentRunResult(
            final_output='first',
            turns=1,
            tool_calls=0,
            transcript=(),
            session_id=session_id,
            session_path=str(self.runtime_config.session_directory / f'{session_id}.json'),
        )

    def resume(self, prompt: str, stored_session: StoredAgentSession, *, event_callback=None) -> AgentRunResult:
        self.prompts.append(('resume', prompt))
        return AgentRunResult(
            final_output=f'resumed {stored_session.session_id}',
            turns=1,
            tool_calls=0,
            transcript=(),
            session_id=stored_session.session_id,
        )


class TuiAppTests(unittest.TestCase):
    def test_controller_continues_session_across_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            agents: list[FakeAgent] = []

            def factory(_model_config: ModelConfig, runtime_config: AgentRuntimeConfig) -> FakeAgent:
                agent = FakeAgent(runtime_config)
                agents.append(agent)
                return agent

            controller = TuiController(
                model_config=ModelConfig(model='fake'),
                runtime_config=AgentRuntimeConfig(
                    cwd=workspace,
                    session_directory=workspace / '.port_sessions' / 'agent',
                    scratchpad_root=workspace / '.port_sessions' / 'scratchpad',
                ),
                agent_factory=factory,
            )

            first = controller.run_prompt('hello')
            second = controller.run_prompt('continue')

        self.assertEqual(first.session_id, 'session-one')
        self.assertEqual(second.final_output, 'resumed session-one')
        self.assertEqual(agents[0].prompts, [('run', 'hello')])
        self.assertEqual(agents[1].prompts, [('resume', 'continue')])

    def test_controller_lists_recent_sessions_newest_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            session_dir = workspace / '.port_sessions' / 'agent'
            controller = TuiController(
                model_config=ModelConfig(model='fake'),
                runtime_config=AgentRuntimeConfig(cwd=workspace, session_directory=session_dir),
            )
            save_agent_session(
                StoredAgentSession(
                    session_id='older-session',
                    model_config={'model': 'fake'},
                    runtime_config={'cwd': str(workspace)},
                    system_prompt_parts=(),
                    user_context={},
                    system_context={},
                    messages=({'role': 'user', 'content': 'older prompt'},),
                    turns=1,
                    tool_calls=0,
                    usage={'total_tokens': 10},
                    total_cost_usd=0.0,
                    file_history=(),
                ),
                directory=session_dir,
            )
            save_agent_session(
                StoredAgentSession(
                    session_id='newer-session',
                    model_config={'model': 'fake'},
                    runtime_config={'cwd': str(workspace)},
                    system_prompt_parts=(),
                    user_context={},
                    system_context={},
                    messages=({'role': 'user', 'content': 'newer prompt'},),
                    turns=2,
                    tool_calls=3,
                    usage={'total_tokens': 20},
                    total_cost_usd=0.0,
                    file_history=(),
                ),
                directory=session_dir,
            )
            older_path = session_dir / 'older-session.json'
            newer_path = session_dir / 'newer-session.json'
            older_path.touch()
            newer_path.touch()

            summaries = controller.recent_sessions(limit=5)

        self.assertEqual([summary.session_id for summary in summaries], ['newer-session', 'older-session'])
        self.assertEqual(summaries[0].preview, 'newer prompt')
        self.assertEqual(summaries[0].turns, 2)
        self.assertEqual(summaries[0].tool_calls, 3)

    def test_controller_empty_filtered_picker_does_not_resume_unfiltered_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            session_dir = workspace / '.port_sessions' / 'agent'
            save_agent_session(
                StoredAgentSession(
                    session_id='real-session',
                    model_config={'model': 'fake'},
                    runtime_config={'cwd': str(workspace)},
                    system_prompt_parts=(),
                    user_context={},
                    system_context={},
                    messages=({'role': 'user', 'content': 'Existing prompt'},),
                    turns=1,
                    tool_calls=0,
                    usage={},
                    total_cost_usd=0.0,
                    file_history=(),
                ),
                directory=session_dir,
            )
            controller = TuiController(
                model_config=ModelConfig(model='fake'),
                runtime_config=AgentRuntimeConfig(cwd=workspace, session_directory=session_dir),
            )

            rendered = controller.format_recent_sessions(filter_text='no-match')
            resumed = controller.resume_recent_session('1')

        self.assertIn('No saved sessions found matching', rendered)
        self.assertIsNone(resumed)
        self.assertIsNone(controller.session_id)

    def test_controller_filters_sessions_and_resumes_from_last_picker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            session_dir = workspace / '.port_sessions' / 'agent'
            for session_id, prompt in (('alpha-session', 'Build the alpha feature'), ('beta-session', 'Fix the beta bug')):
                save_agent_session(
                    StoredAgentSession(
                        session_id=session_id,
                        model_config={'model': 'fake'},
                        runtime_config={'cwd': str(workspace)},
                        system_prompt_parts=(),
                        user_context={},
                        system_context={},
                        messages=({'role': 'user', 'content': prompt},),
                        turns=1,
                        tool_calls=0,
                        usage={},
                        total_cost_usd=0.0,
                        file_history=(),
                    ),
                    directory=session_dir,
                )
            controller = TuiController(
                model_config=ModelConfig(model='fake'),
                runtime_config=AgentRuntimeConfig(cwd=workspace, session_directory=session_dir),
            )

            rendered = controller.format_recent_sessions(filter_text='beta')
            resumed = controller.resume_recent_session('1')

        self.assertIn('1. beta-session', rendered)
        self.assertNotIn('alpha-session', rendered)
        self.assertEqual(resumed, 'beta-session')
        self.assertEqual(controller.session_id, 'beta-session')

    def test_controller_resumes_recent_session_by_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            session_dir = workspace / '.port_sessions' / 'agent'
            controller = TuiController(
                model_config=ModelConfig(model='fake'),
                runtime_config=AgentRuntimeConfig(cwd=workspace, session_directory=session_dir),
            )
            save_agent_session(
                StoredAgentSession(
                    session_id='indexed-session',
                    model_config={'model': 'fake'},
                    runtime_config={'cwd': str(workspace)},
                    system_prompt_parts=(),
                    user_context={},
                    system_context={},
                    messages=({'role': 'user', 'content': 'indexed prompt'},),
                    turns=1,
                    tool_calls=0,
                    usage={},
                    total_cost_usd=0.0,
                    file_history=(),
                ),
                directory=session_dir,
            )

            resumed = controller.resume_recent_session('1')

        self.assertEqual(resumed, 'indexed-session')
        self.assertEqual(controller.session_id, 'indexed-session')

    def test_controller_agents_report_shows_main_and_delegate_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='main-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace, delegate_model='helper-model', delegate_base_url='http://127.0.0.1:8081/v1'),
            )

            report = controller.agents_report()

        self.assertIn('# Agents', report)
        self.assertIn('Main model: main-model', report)
        self.assertIn('Delegate model: helper-model', report)
        self.assertIn('Delegate backend: http://127.0.0.1:8081/v1', report)

    def test_controller_diff_report_handles_non_git_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='fake'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )

            report = controller.diff_report()

        self.assertIn('# Diff', report)
        self.assertIn('not inside a git worktree', report)

    def test_controller_diff_report_includes_staged_and_untracked_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            subprocess.run(['git', 'init'], cwd=workspace, check=True, capture_output=True, text=True)
            subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=workspace, check=True)
            subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=workspace, check=True)
            tracked = workspace / 'tracked.txt'
            tracked.write_text('base\n', encoding='utf-8')
            subprocess.run(['git', 'add', 'tracked.txt'], cwd=workspace, check=True)
            subprocess.run(['git', 'commit', '-m', 'initial'], cwd=workspace, check=True, capture_output=True, text=True)
            tracked.write_text('base\nstaged change\n', encoding='utf-8')
            subprocess.run(['git', 'add', 'tracked.txt'], cwd=workspace, check=True)
            untracked = workspace / 'new.txt'
            untracked.write_text('untracked content\n', encoding='utf-8')
            controller = TuiController(
                model_config=ModelConfig(model='fake'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )

            report = controller.diff_report()

        self.assertIn('tracked.txt', report)
        self.assertIn('staged change', report)
        self.assertIn('new.txt', report)
        self.assertIn('untracked content', report)

    def test_controller_diff_report_shows_changed_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            subprocess.run(['git', 'init'], cwd=workspace, check=True, capture_output=True, text=True)
            subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=workspace, check=True)
            subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=workspace, check=True)
            target = workspace / 'hello.txt'
            target.write_text('hello\n', encoding='utf-8')
            subprocess.run(['git', 'add', 'hello.txt'], cwd=workspace, check=True)
            subprocess.run(['git', 'commit', '-m', 'initial'], cwd=workspace, check=True, capture_output=True, text=True)
            target.write_text('hello\nchanged\n', encoding='utf-8')
            controller = TuiController(
                model_config=ModelConfig(model='fake'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )

            report = controller.diff_report()

        self.assertIn('# Diff', report)
        self.assertIn('hello.txt', report)
        self.assertIn('changed', report)

    def test_controller_renders_local_agent_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='fake'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )

            tools = controller.tools_report()
            permissions = controller.permissions_report()
            context = controller.context_report()

        self.assertIn('# Tools', tools)
        self.assertIn('`web_search`', tools)
        self.assertIn('# Permissions', permissions)
        self.assertIn('Shell commands: disabled', permissions)
        self.assertIn('# Context', context)


    def test_permission_toggles_update_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='fake'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )

            controller.toggle_write()
            controller.toggle_shell()
            controller.toggle_unsafe()

        permissions = controller.runtime_config.permissions
        self.assertTrue(permissions.allow_file_write)
        self.assertTrue(permissions.allow_shell_commands)
        self.assertTrue(permissions.allow_destructive_shell_commands)

    def test_delegate_events_render_plainly(self) -> None:
        rendered = render_plain_event(
            {
                'type': 'delegate_subtask_result',
                'label': 'scan',
                'session_id': 'child-session',
                'stop_reason': 'stop',
            }
        )

        self.assertIn('delegate_subtask', rendered)
        self.assertIn('child-session', rendered)

    def test_result_display_format_summarizes_tools_and_normalizes_assistant(self) -> None:
        assistant, tool_summary, delegate_summary = _format_result_for_display(
            AgentRunResult(
                final_output='assistant\n<think>\n\n</think>\n\nDone.',
                turns=1,
                tool_calls=1,
                transcript=(),
                events=(
                    {'type': 'tool_start', 'tool_name': 'list_dir', 'tool_call_id': 'a'},
                    {'type': 'tool_delta', 'tool_name': 'list_dir', 'delta': 'noisy'},
                    {'type': 'tool_result', 'tool_name': 'list_dir', 'ok': True},
                ),
            )
        )

        self.assertEqual(assistant, 'Done.')
        self.assertEqual(tool_summary, 'list_dir ok')
        self.assertEqual(delegate_summary, '')


    def test_assistant_text_normalization_removes_role_prefix(self) -> None:
        self.assertEqual(
            _normalize_assistant_text('assistant\nHello there.'),
            'Hello there.',
        )
        self.assertEqual(
            _normalize_assistant_text('\nAssistant:\nReady.'),
            'Ready.',
        )

    def test_assistant_text_normalization_removes_empty_think_block(self) -> None:
        self.assertEqual(
            _normalize_assistant_text('assistant\n<think>\n\n</think>\n\nVisible.'),
            'Visible.',
        )

    def test_curses_renderer_preserves_markdown_line_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='fake'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )
            tui = _CursesTui(controller)
            tui.lines = []
            tui._append('assistant', '# Heading\n\n- first item\n- second item')

            rendered = [line for _role, line in tui._render_lines(80)]

        joined = '\n'.join(rendered)
        self.assertIn('ASSISTANT | Heading', joined)
        self.assertIn('| ', joined)
        self.assertIn('- first item', joined)
        self.assertIn('- second item', joined)

    def test_plain_handler_supports_filtered_sessions_and_empty_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            session_dir = workspace / '.port_sessions' / 'agent'
            save_agent_session(
                StoredAgentSession(
                    session_id='real-session',
                    model_config={'model': 'fake'},
                    runtime_config={'cwd': str(workspace)},
                    system_prompt_parts=(),
                    user_context={},
                    system_context={},
                    messages=({'role': 'user', 'content': 'Existing prompt'},),
                    turns=1,
                    tool_calls=0,
                    usage={},
                    total_cost_usd=0.0,
                    file_history=(),
                ),
                directory=session_dir,
            )
            controller = TuiController(
                model_config=ModelConfig(model='fake'),
                runtime_config=AgentRuntimeConfig(cwd=workspace, session_directory=session_dir),
            )

            sessions_handled = _handle_plain_command(controller, '/sessions no-match')
            resume_handled = _handle_plain_command(controller, '/resume 1')

        self.assertEqual(sessions_handled, 'handled')
        self.assertEqual(resume_handled, 'handled')
        self.assertIsNone(controller.session_id)

    def test_plain_handler_supports_advertised_local_report_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='fake'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )
            commands = ['/tools', '/permissions', '/context', '/sessions']

            handled = [_handle_plain_command(controller, command) for command in commands]

        self.assertEqual(handled, ['handled', 'handled', 'handled', 'handled'])

    def test_rich_handler_supports_advertised_local_report_commands(self) -> None:
        class FakeConsole:
            def __init__(self) -> None:
                self.rendered: list[object] = []

            def print(self, value: object) -> None:
                self.rendered.append(value)

        class FakePanel:
            def __init__(self, body: object, *, title: str | None = None) -> None:
                self.body = body
                self.title = title

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            controller = TuiController(
                model_config=ModelConfig(model='fake'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )
            console = FakeConsole()
            commands = ['/tools', '/permissions', '/context', '/sessions']

            handled = [_handle_rich_command(console, FakePanel, controller, command) for command in commands]

        self.assertEqual(handled, ['handled', 'handled', 'handled', 'handled'])
        self.assertGreaterEqual(len(console.rendered), 4)

    def test_wrap_display_line_indents_bullet_continuations(self) -> None:
        wrapped = _wrap_display_line('- Bullet item with enough text to wrap onto the next line', 35)

        self.assertGreater(len(wrapped), 1)
        self.assertTrue(wrapped[1].startswith('  '))

    def test_wrap_display_line_indents_ordered_list_continuations(self) -> None:
        wrapped = _wrap_display_line('12. Ordered item with enough text to wrap onto the next line', 35)

        self.assertGreater(len(wrapped), 1)
        self.assertTrue(wrapped[1].startswith('    '))


if __name__ == '__main__':
    unittest.main()
