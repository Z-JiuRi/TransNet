#!/bin/bash

PID=$1

if [ -z "$PID" ]; then
    echo "用法: $0 <PID>"
    exit 1
fi

# 查找 stdout/stderr 重定向的文件
LOG_FILE=$(sudo lsof -p $PID 2>/dev/null | grep -E " [12][uw] " | grep REG | awk '{print $NF}' | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "未找到日志文件，可能输出到终端或管道"
else
    echo "找到日志文件: $LOG_FILE"
    echo "开始实时跟踪（Ctrl+C 退出）..."
    tail -f "$LOG_FILE"
fi