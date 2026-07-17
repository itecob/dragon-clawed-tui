from pathlib import Path

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentRuntimeConfig, ModelConfig


def test_prompt_too_long_detector_handles_llama_server_available_context_error(tmp_path: Path) -> None:
    agent = LocalCodingAgent(
        model_config=ModelConfig(model='fake'),
        runtime_config=AgentRuntimeConfig(cwd=tmp_path),
    )

    assert agent._is_prompt_too_long_error(
        RuntimeError('request (8647 tokens) exceeds the available context size (8192 tokens), try increasing it')
    )
