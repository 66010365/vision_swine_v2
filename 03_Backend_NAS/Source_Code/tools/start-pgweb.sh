#!/bin/bash

if ! pgrep -f "/home/dev/vision-swine/tools/pgweb_vision_swine" > /dev/null; then
    nohup /home/dev/vision-swine/tools/pgweb_vision_swine \
        --bind 127.0.0.1 \
        --listen 8082 \
        --skip-open \
        > /home/dev/vision-swine/logs/pgweb.log 2>&1 &
fi
