from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentRuntimeConfig, ModelConfig
from src.session_store import load_agent_session, serialize_model_config


class TimeoutStreamingHTTPResponse:
    def readline(self) -> bytes:
        raise TimeoutError('stream stalled')

    def __enter__(self) -> 'TimeoutStreamingHTTPResponse':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def timeout_urlopen(request_obj, timeout=None):  # noqa: ANN001
    return TimeoutStreamingHTTPResponse()


class AgentAcceptanceTests(unittest.TestCase):
    def test_stream_timeout_returns_backend_error_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            with patch('src.openai_compat.request.urlopen', side_effect=timeout_urlopen):
                agent = LocalCodingAgent(
                    model_config=ModelConfig(
                        model='fake-model',
                        base_url='http://127.0.0.1:8000/v1',
                        timeout_seconds=0.01,
                    ),
                    runtime_config=AgentRuntimeConfig(
                        cwd=workspace,
                        stream_model_responses=True,
                    ),
                )
                result = agent.run('hello')

        self.assertEqual(result.stop_reason, 'backend_error')
        self.assertIn('timed out', result.final_output.lower())
        self.assertFalse(result.session_path and 'secret-key' in Path(result.session_path).read_text())

    def test_persisted_agent_session_redacts_api_key(self) -> None:
        responses = [
            {
                'choices': [
                    {
                        'message': {'role': 'assistant', 'content': 'Done.'},
                        'finish_reason': 'stop',
                    }
                ],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
            }
        ]

        class FakeHTTPResponse:
            def __init__(self, payload: dict[str, object]) -> None:
                self.payload = payload

            def read(self) -> bytes:
                return json.dumps(self.payload).encode('utf-8')

            def __enter__(self) -> 'FakeHTTPResponse':
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

        queued = [FakeHTTPResponse(payload) for payload in responses]

        def fake_urlopen(request_obj, timeout=None):  # noqa: ANN001
            return queued.pop(0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            session_dir = workspace / 'sessions'
            with patch('src.openai_compat.request.urlopen', side_effect=fake_urlopen):
                agent = LocalCodingAgent(
                    model_config=ModelConfig(
                        model='fake-model',
                        base_url='http://127.0.0.1:8000/v1',
                        api_key='secret-key-that-must-not-persist',
                    ),
                    runtime_config=AgentRuntimeConfig(cwd=workspace, session_directory=session_dir),
                )
                result = agent.run('hello')

            assert result.session_id is not None
            raw_session = (session_dir / f'{result.session_id}.json').read_text(encoding='utf-8')
            stored = load_agent_session(result.session_id, directory=session_dir)

        self.assertNotIn('secret-key-that-must-not-persist', raw_session)
        self.assertEqual(stored.model_config.get('api_key'), '<redacted>')
        self.assertEqual(serialize_model_config(ModelConfig(model='fake', api_key='real'))['api_key'], '<redacted>')


if __name__ == '__main__':
    unittest.main()
