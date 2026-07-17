<p align="center">
  <img src="images/logo.png" alt="Dragon Clawed TUI logo" width="360" />
</p>

<h1 align="center">Dragon Clawed TUI</h1>

<p align="center">
  <em>A Python coding-agent TUI for local and OpenAI-compatible models.</em>
</p>

---

## About

Dragon Clawed TUI is a terminal coding-agent interface built around a Python agent runtime. It provides a Textual-based TUI, an OpenAI-compatible model client, local file/code tools, session persistence, permission controls, delegated sub-agents, and optional local voice input.

The project is designed to keep the same OpenAI-compatible connection flexibility as the CLI runtime: llama.cpp/llama-server, Ollama, vLLM, LiteLLM, local gateways, or remote OpenAI-compatible providers can be selected with environment variables or CLI flags.

## Current Features

- Textual TUI with themed transcript, status bar, activity updates, and multiline prompt composer
- Local coding-agent loop with read, write, edit, search, web search, shell, and delegation tools
- Permission tiers for read-only, write, shell, and unsafe actions
- Session persistence and resume support
- Transcript copy/export and local search
- Main/helper model routing for delegated agents
- Context compaction and malformed-history recovery for llama.cpp-compatible backends
- Optional push-to-talk voice transcription via Faster Whisper
- Desktop launcher script that can either use an existing OpenAI-compatible backend or optionally start configured local llama-server services
- Unit and regression tests for runtime, TUI, launcher, tools, voice, and session behavior

## Requirements

- Python 3.10+
- A terminal with good TUI support
- An OpenAI-compatible chat completions backend
- Optional for voice input: `arecord` and Faster Whisper dependencies

## Quick Start

Create a virtual environment and install the package in editable mode:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Configure an OpenAI-compatible backend:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=local-token
export OPENAI_MODEL=your-model-id
```

Run the TUI:

```bash
python -m src.main tui --cwd .
```

Run a one-shot agent prompt:

```bash
python -m src.main agent "Inspect this repository and summarize the test layout." --cwd .
```


### Supported Backend Configuration

The runtime uses the standard OpenAI-compatible variables:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=local-token
export OPENAI_MODEL=your-model-id
```

You can also pass backend settings directly:

```bash
python -m src.main tui --model your-model-id --base-url http://127.0.0.1:8000/v1 --api-key local-token
```

Common backend choices:

- llama.cpp / llama-server
- Ollama OpenAI-compatible endpoint
- vLLM OpenAI-compatible server
- LiteLLM Proxy
- Remote OpenAI-compatible providers

## Local Backend Notes

For local llama.cpp/llama-server backends, use a dummy bearer token such as `local-token`. Do not send provider API keys to local backends.

The included desktop launcher does not require a specific model. If `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL` point at an existing backend, it can launch the TUI directly. If `CLAWED_MAIN_MODEL` or `CLAWED_HELPER_MODEL` point at local GGUF files and `CLAWED_LLAMA_SERVER_BIN` is available, it can optionally start llama-server services for those models.

## Voice Input

Voice input is optional. The TUI uses F4 as a push-to-talk toggle when voice dependencies and audio hardware are available. Transcription is performed locally when configured with Faster Whisper.

## Development

Run the test suite:

```bash
python -m pytest -q
```

Run focused TUI/runtime tests:

```bash
python -m pytest -q tests/test_textual_tui_app.py tests/test_agent_runtime.py tests/test_tui_app.py
```

## Security

- Keep real provider keys out of the repository.
- Use `.env` only for local, uncommitted configuration.
- Prefer dummy tokens for local model servers.
- Review shell and unsafe permissions before enabling them.

## License

MIT License. See [LICENSE](LICENSE).
