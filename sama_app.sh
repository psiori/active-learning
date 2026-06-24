#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
crid_repo_dir="${AUTOCRANE_CLOUD_PATH:-${script_dir}/../autocrane-cloud}"
repo_dir="$(cd "${crid_repo_dir}/apps/crid" && pwd)"
frontend_dir="${script_dir}/sama_frontend"
session_name="${TMUX_SESSION_NAME:-sama}"
backend_host="${BACKEND_HOST:-127.0.0.1}"
backend_port="${BACKEND_PORT:-8080}"
frontend_host="${FRONTEND_HOST:-127.0.0.1}"
frontend_port="${FRONTEND_PORT:-5175}"

wrap_cmd() {
    local command="$1"
    printf "%s; exit_code=\$?; echo; echo \"[process exited with code \$exit_code]\"; echo \"Pane kept open by tmux remain-on-exit.\"" "${command}"
}

backend_prelude() {
    cat <<EOF
existing_pids=\$(lsof -tiTCP:${backend_port} -sTCP:LISTEN 2>/dev/null || true)
if [[ -n "\${existing_pids}" ]]; then
    echo "Killing stale backend listener(s) on port ${backend_port}: \${existing_pids}"
    kill -9 \${existing_pids} 2>/dev/null || true
    sleep 1
fi
EOF
}

if ! command -v tmux >/dev/null 2>&1; then
    echo "Missing required command: tmux"
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "Missing required command: uv"
    exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "No active virtual environment detected."
    echo "Start this from the CRID shell created by ./run_local_uv.sh ... --shell"
    exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "Missing required command: npm"
    exit 1
fi

if [[ ! -d "${frontend_dir}/node_modules" ]]; then
    echo "Sama frontend dependencies are not installed."
    echo "Run: cd ${frontend_dir} && npm install"
    exit 1
fi

backend_cmd="$(wrap_cmd "$(backend_prelude)
cd ${repo_dir} && PYTHONPATH=${repo_dir} BACKEND_HOST=${backend_host} BACKEND_PORT=${backend_port} uv run --active --project ${script_dir} --extra app --extra crid uvicorn active_learning.api.main:app --reload --host ${backend_host} --port ${backend_port}")"
frontend_cmd="$(wrap_cmd "cd ${frontend_dir} && FRONTEND_HOST=${frontend_host} FRONTEND_PORT=${frontend_port} VITE_API_BASE_URL=http://${backend_host}:${backend_port} npm run dev -- --host ${frontend_host} --port ${frontend_port}")"

if tmux has-session -t "${session_name}" 2>/dev/null; then
    tmux attach-session -t "${session_name}"
    exit 0
fi

tmux new-session -d -s "${session_name}" -c "${repo_dir}" "bash -c '${backend_cmd}'"
tmux set-option -t "${session_name}" remain-on-exit on
tmux split-window -v -t "${session_name}:0.0" -c "${frontend_dir}" "bash -c '${frontend_cmd}'"
tmux select-pane -t "${session_name}:0.0"
tmux select-layout -t "${session_name}:0" even-vertical

echo "tmux session: ${session_name}"
echo "Backend: http://${backend_host}:${backend_port}"
echo "Sama frontend: http://${frontend_host}:${frontend_port}"

tmux attach-session -t "${session_name}"
