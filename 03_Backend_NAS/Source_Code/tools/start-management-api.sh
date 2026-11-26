
#!/bin/bash

# ======================================================
# Application Configuration
# ======================================================

PROJECT_ROOT="/home/dev/vision-swine"
BACKEND_DIR="${PROJECT_ROOT}/management_system/backend"
VENV_PYTHON="${BACKEND_DIR}/venv/bin/python"
UVICORN="${BACKEND_DIR}/venv/bin/uvicorn"
APP_MODULE="vision_swine_api.main:app"
HOST="127.0.0.1"
PORT="8002"
LOG_FILE="${PROJECT_ROOT}/logs/management-api.log"


# ======================================================
# Startup Validation
# ======================================================

if [[ ! -x "${UVICORN}" ]]; then
    echo "ERROR: Uvicorn executable not found: ${UVICORN}"
    exit 1
fi

if [[ ! -f "${BACKEND_DIR}/vision_swine_api/main.py" ]]; then
    echo "ERROR: Application entry point not found."
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs"


# ======================================================
# Existing Process Check
# ======================================================

if pgrep -f "[u]vicorn ${APP_MODULE}.*--port ${PORT}" > /dev/null; then
    echo "Management API is already running on port ${PORT}."
    exit 0
fi


# ======================================================
# Start Management API
# ======================================================

cd "${BACKEND_DIR}" || exit 1

nohup "${UVICORN}" "${APP_MODULE}" \
    --host "${HOST}" \
    --port "${PORT}" \
    >> "${LOG_FILE}" 2>&1 &

API_PID=$!


# ======================================================
# Startup Result
# ======================================================

sleep 2

if kill -0 "${API_PID}" 2>/dev/null; then
    echo "Management API startup command completed."
    echo "PID: ${API_PID}"
    echo "Address: http://${HOST}:${PORT}"
    echo "Log: ${LOG_FILE}"
else
    echo "ERROR: Management API failed to start."
    echo "Check log: ${LOG_FILE}"
    exit 1
fi
