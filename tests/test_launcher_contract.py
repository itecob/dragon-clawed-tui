from __future__ import annotations

from pathlib import Path


def test_desktop_launcher_does_not_set_default_cumulative_input_budget() -> None:
    launcher = Path('scripts/launch_clawed_tui_desktop.sh').read_text(encoding='utf-8')
    assert 'CLAWED_MAX_INPUT_TOKENS:-7900' not in launcher
    assert '--max-input-tokens "$CLAWED_MAX_INPUT_TOKENS"' in launcher



def test_desktop_launcher_does_not_enable_shell_by_default() -> None:
    launcher = Path('scripts/launch_clawed_tui_desktop.sh').read_text(encoding='utf-8')
    tui_args_block = launcher.split('TUI_ARGS=(')[1].split(')')[0]
    assert '--allow-write' in tui_args_block
    assert '--allow-shell' not in tui_args_block
    assert 'CLAWED_TUI_ALLOW_SHELL' in launcher

def test_desktop_launcher_uses_long_local_model_timeout_by_default() -> None:
    launcher = Path('scripts/launch_clawed_tui_desktop.sh').read_text(encoding='utf-8')
    assert 'CLAWED_MODEL_TIMEOUT_SECONDS:-300' in launcher
    assert 'CLAWED_MODEL_TIMEOUT_SECONDS:-60' not in launcher


def test_desktop_launcher_context_defaults_leave_room_for_tool_schemas() -> None:
    launcher = Path('scripts/launch_clawed_tui_desktop.sh').read_text(encoding='utf-8')
    assert 'CLAWED_AUTO_COMPACT_THRESHOLD:-10000' in launcher
    assert 'CLAWED_AUTO_SNIP_THRESHOLD:-13000' in launcher

def test_desktop_launcher_limits_local_server_parallelism_by_default() -> None:
    launcher = Path('scripts/launch_clawed_tui_desktop.sh').read_text(encoding='utf-8')
    assert 'CLAWED_SERVER_PARALLEL:-1' in launcher
    assert '--parallel' in launcher
    assert 'CLAWED_SERVER_THREADS_HTTP:-4' in launcher
    assert '--threads-http' in launcher

def test_project_env_does_not_contain_groq_api_key() -> None:
    env_path = Path('.env')
    if not env_path.exists():
        return
    env_text = env_path.read_text(encoding='utf-8')
    assert 'gsk_' not in env_text


def test_desktop_launcher_loads_repo_env_without_source() -> None:
    launcher = Path('scripts/launch_clawed_tui_desktop.sh').read_text(encoding='utf-8')
    assert 'load_env_file "$REPO_DIR/.env"' in launcher
    assert 'source "$REPO_DIR/.env"' not in launcher


def test_desktop_launcher_does_not_send_secret_file_key_to_local_backend() -> None:
    launcher = Path('scripts/launch_clawed_tui_desktop.sh').read_text(encoding='utf-8')
    local_backend_block = launcher.split('export OPENAI_BASE_URL="http://127.0.0.1:${MAIN_PORT}/v1"', 1)[1].split('export OPENAI_MODEL=', 1)[0]
    assert 'CLAWED_ORIGINAL_ENV[OPENAI_API_KEY]' in local_backend_block
    assert 'export OPENAI_API_KEY="local-token"' in local_backend_block
    assert 'export OPENAI_API_KEY="${OPENAI_API_KEY:-local-token}"' in local_backend_block
