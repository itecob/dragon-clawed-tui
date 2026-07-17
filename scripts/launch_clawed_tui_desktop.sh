#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="$REPO_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

declare -A CLAWED_ORIGINAL_ENV=()
while IFS='=' read -r key _value; do
  if [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    CLAWED_ORIGINAL_ENV["$key"]=1
  fi
done < <(env)

load_env_file() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#${line%%[![:space:]]*}}"
    line="${line%${line##*[![:space:]]}}"
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
    [[ "$line" == *"="* ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    key="${key%${key##*[![:space:]]}}"
    key="${key#${key%%[![:space:]]*}}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ -n "${CLAWED_ORIGINAL_ENV[$key]+x}" ]]; then
      continue
    fi
    value="${value#${value%%[![:space:]]*}}"
    value="${value%${value##*[![:space:]]}}"
    if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
      value="${value:1:${#value}-2}"
    fi
    export "$key=$value"
  done < "$file"
}

load_env_file "$REPO_DIR/.env.ollama"
load_env_file "$REPO_DIR/.env"

SERVICE_DIR="${XDG_RUNTIME_DIR:-/tmp}/dragon-clawed-tui"
mkdir -p "$SERVICE_DIR"

MAIN_PORT="${CLAWED_MAIN_PORT:-8080}"
HELPER_PORT="${CLAWED_HELPER_PORT:-8081}"
MAIN_MODEL="${CLAWED_MAIN_MODEL:-}"
HELPER_MODEL="${CLAWED_HELPER_MODEL:-}"
LLAMA_SERVER_BIN="${CLAWED_LLAMA_SERVER_BIN:-}"
WORKSPACE="${CLAWED_WORKSPACE:-$REPO_DIR}"

if [[ -z "$LLAMA_SERVER_BIN" ]]; then
  for candidate in \
    "$REPO_DIR/llama.cpp/build/bin/llama-server" \
    "$HOME/llama.cpp/build/bin/llama-server" \
    "llama-server"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      LLAMA_SERVER_BIN="$(command -v "$candidate")"
      break
    fi
    if [[ -x "$candidate" ]]; then
      LLAMA_SERVER_BIN="$candidate"
      break
    fi
  done
fi

port_pid() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | head -n 1
    return 0
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp "sport = :$port" 2>/dev/null \
      | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
      | head -n 1
  fi
}

describe_pid() {
  local pid="$1"
  ps -p "$pid" -o pid=,comm=,args= 2>/dev/null || true
}

service_ready() {
  local port="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1
    return $?
  fi
  [[ -n "$(port_pid "$port" || true)" ]]
}

prompt_for_existing_service() {
  local label="$1"
  local port="$2"
  local pid="$3"

  echo
  echo "Detected an existing $label service on port $port:"
  describe_pid "$pid"
  echo
  echo "Options:"
  echo "  [l] Leave it running and use it"
  echo "  [s] Gracefully stop it, then start the configured $label service"
  echo "  [a] Abort startup"
  while true; do
    read -r -p "Choose l/s/a: " choice
    case "${choice,,}" in
      l|leave)
        return 0
        ;;
      s|stop)
        echo "Stopping PID $pid..."
        kill "$pid" 2>/dev/null || true
        for _ in {1..20}; do
          if ! kill -0 "$pid" 2>/dev/null; then
            return 1
          fi
          sleep 0.25
        done
        echo "PID $pid did not exit after SIGTERM."
        read -r -p "Force kill it? [y/N] " force
        if [[ "${force,,}" == "y" || "${force,,}" == "yes" ]]; then
          kill -KILL "$pid" 2>/dev/null || true
          return 1
        fi
        return 0
        ;;
      a|abort)
        echo "Aborted."
        exit 1
        ;;
      *)
        echo "Please choose l, s, or a."
        ;;
    esac
  done
}

start_llama_service() {
  local label="$1"
  local port="$2"
  local model="$3"
  local gpu="$4"
  local pid_file="$SERVICE_DIR/${label}.pid"
  local log_file="$SERVICE_DIR/${label}.log"

  local existing_pid
  existing_pid="$(port_pid "$port" || true)"
  if [[ -n "$existing_pid" ]]; then
    if [[ -f "$pid_file" && "$(cat "$pid_file" 2>/dev/null || true)" == "$existing_pid" ]]; then
      echo "$label is already running on port $port (PID $existing_pid)."
      return 0
    fi
    if prompt_for_existing_service "$label" "$port" "$existing_pid"; then
      return 0
    fi
  fi

  if [[ -z "$model" ]]; then
    echo "No model configured for $label; skipping local service start."
    return 0
  fi
  if [[ -z "$LLAMA_SERVER_BIN" || ! -x "$LLAMA_SERVER_BIN" ]]; then
    echo "No executable llama-server found; cannot start $label."
    echo "Set CLAWED_LLAMA_SERVER_BIN in $REPO_DIR/.env."
    return 0
  fi
  if [[ ! -f "$model" ]]; then
    echo "Configured model for $label does not exist: $model"
    return 0
  fi

  local model_id="$label"
  local context_size="${CLAWED_CONTEXT_SIZE:-4096}"
  case "$label" in
    main)
      model_id="${CLAWED_MAIN_MODEL_ID:-${OPENAI_MODEL:-local-model}}"
      context_size="${CLAWED_MAIN_CONTEXT_SIZE:-$context_size}"
      ;;
    helper)
      model_id="${CLAWED_HELPER_MODEL_ID:-${CLAWED_DELEGATE_MODEL:-$model_id}}"
      context_size="${CLAWED_HELPER_CONTEXT_SIZE:-$context_size}"
      ;;
  esac
  local extra_args=()
  if [[ -n "${CLAWED_SERVER_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${CLAWED_SERVER_EXTRA_ARGS})
  fi

  echo "Starting $label on port $port..."
  echo "  model=$model" >"$log_file"
  echo "  model_id=$model_id" >>"$log_file"
  echo "  context=$context_size gpu=$gpu" >>"$log_file"
  : >"$pid_file"
  setsid -f bash -c 'pid_file="$1"; shift; echo "$$" > "$pid_file"; exec "$@"' \
    bash "$pid_file" \
    env CUDA_VISIBLE_DEVICES="$gpu" "$LLAMA_SERVER_BIN" \
      -m "$model" \
      --host 127.0.0.1 \
      --port "$port" \
      -ngl "${CLAWED_N_GPU_LAYERS:-99}" \
      -c "$context_size" \
      --no-ui \
      -a "$model_id" \
      --reasoning "${CLAWED_REASONING_MODE:-off}" \
      --log-verbosity "${CLAWED_SERVER_LOG_VERBOSITY:-2}" \
      --parallel "${CLAWED_SERVER_PARALLEL:-1}" \
      --threads-http "${CLAWED_SERVER_THREADS_HTTP:-4}" \
      -ctk "${CLAWED_CACHE_TYPE_K:-q8_0}" \
      -ctv "${CLAWED_CACHE_TYPE_V:-q8_0}" \
      "${extra_args[@]}" \
      </dev/null >>"$log_file" 2>&1

  local pid=""
  for _ in {1..20}; do
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]]; then
      break
    fi
    sleep 0.1
  done
  if [[ -z "$pid" ]]; then
    echo "$label failed to record a service PID. Log:"
    tail -n 80 "$log_file" || true
    return 1
  fi

  for _ in {1..160}; do
    if service_ready "$port"; then
      echo "$label API is ready on port $port (PID $pid)."
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$label exited during startup. Log:"
      tail -n 80 "$log_file" || true
      return 1
    fi
    sleep 0.5
  done

  echo "$label did not become API-ready. Log:"
  tail -n 80 "$log_file" || true
  return 1
}

echo "ClawedCode TUI launcher"
echo "Repo: $REPO_DIR"
echo "Workspace: $WORKSPACE"
echo

MAIN_SERVICE_OK=1
if ! start_llama_service "main" "$MAIN_PORT" "$MAIN_MODEL" "${CLAWED_MAIN_GPU:-0}"; then
  MAIN_SERVICE_OK=0
  echo
  echo "Local main service did not start. You can still open the TUI using the configured OPENAI_BASE_URL: ${OPENAI_BASE_URL:-<unset>}"
  read -r -p "Continue to TUI anyway? [Y/n] " continue_choice
  case "${continue_choice,,}" in
    n|no)
      echo "Aborted."
      exit 1
      ;;
  esac
fi

if [[ "$MAIN_SERVICE_OK" == "1" ]]; then
  if ! start_llama_service "helper" "$HELPER_PORT" "$HELPER_MODEL" "${CLAWED_HELPER_GPU:-1}"; then
    echo
    echo "Local helper service did not start. Continuing without helper service."
  fi
else
  echo "Skipping helper startup because the main local service is unavailable."
fi

MAIN_LISTENING_PID="$(port_pid "$MAIN_PORT" || true)"
if [[ -n "$MAIN_LISTENING_PID" && ( -z "${OPENAI_BASE_URL:-}" || "${OPENAI_BASE_URL}" == "https://api.groq.com/openai/v1" ) ]]; then
  export OPENAI_BASE_URL="http://127.0.0.1:${MAIN_PORT}/v1"
  if [[ -n "${CLAWED_ORIGINAL_ENV[OPENAI_API_KEY]+x}" ]]; then
    export OPENAI_API_KEY="${OPENAI_API_KEY:-local-token}"
  else
    export OPENAI_API_KEY="local-token"
  fi
  export OPENAI_MODEL="${CLAWED_MAIN_MODEL_ID:-${OPENAI_MODEL:-local-model}}"
fi

echo
echo "Opening Dragon Clawed TUI with:"
echo "  OPENAI_BASE_URL=$OPENAI_BASE_URL"
echo "  OPENAI_MODEL=$OPENAI_MODEL"
echo

set +e
TUI_ARGS=(
  --cwd "$WORKSPACE"
  --allow-write
  --timeout-seconds "${CLAWED_MODEL_TIMEOUT_SECONDS:-300}"
  --auto-compact-threshold "${CLAWED_AUTO_COMPACT_THRESHOLD:-10000}"
  --auto-snip-threshold "${CLAWED_AUTO_SNIP_THRESHOLD:-13000}"
  --delegate-base-url "${CLAWED_HELPER_BASE_URL:-http://127.0.0.1:${HELPER_PORT}/v1}"
)
if [[ "${CLAWED_TUI_ALLOW_SHELL:-0}" == "1" || "${CLAWED_TUI_ALLOW_SHELL:-0}" == "true" ]]; then
  TUI_ARGS+=(--allow-shell)
fi
if [[ -n "${CLAWED_MAX_INPUT_TOKENS:-}" ]]; then
  TUI_ARGS+=(--max-input-tokens "$CLAWED_MAX_INPUT_TOKENS")
fi
if [[ -n "${CLAWED_MAX_TOTAL_TOKENS:-}" ]]; then
  TUI_ARGS+=(--max-total-tokens "$CLAWED_MAX_TOTAL_TOKENS")
fi

PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m src.main tui "${TUI_ARGS[@]}"
TUI_STATUS=$?
set -e

echo
if [[ "$TUI_STATUS" -eq 0 ]]; then
  echo "TUI exited. Local services started by this launcher are left running."
else
  echo "TUI exited with status $TUI_STATUS. The terminal is being kept open for diagnostics."
fi
echo "To close services later, rerun this launcher and choose stop for existing services, or kill PIDs in $SERVICE_DIR."
exec "${SHELL:-bash}" -l
