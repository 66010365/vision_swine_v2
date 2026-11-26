#!/bin/bash

# ==========================================
# Vision Swine - Start All Services
# ==========================================

PROJECT_ROOT="/home/dev/vision-swine"
TOOLS_DIR="${PROJECT_ROOT}/tools"

# Start pgweb
"${TOOLS_DIR}/start-pgweb.sh"

# Start Management API
"${TOOLS_DIR}/start-management-api.sh"
