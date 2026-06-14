#!/bin/bash

# 设置你的 Server酱 SendKey（替换成你自己的）
SENDKEY="SCT361014TFusLJ8YXLo2RuLf6GiKqYWEr"

# 日志文件
LOG_FILE="./monitor_waird.log"

# 循环间隔（秒）
INTERVAL=300  # 5分钟 = 300秒

echo "$(date): 监控脚本已启动，每5分钟检查一次" >> $LOG_FILE

# 无限循环
while true; do
    # 检查进程数量
    COUNT=$(pgrep -f "python.*main.py.*WAIRD" | wc -l)
    
    # 记录日志
    echo "$(date): 检测到 $COUNT 个 WAIRD 进程" >> $LOG_FILE
    
    # 如果进程数为0，发送通知
    if [ $COUNT -eq 0 ]; then
        # 获取主机名
        HOSTNAME=$(hostname)
        
        # 发送消息到 Server酱
        curl -s "https://sctapi.ftqq.com/${SENDKEY}.send" \
            -d "title=${HOSTNAME} - WAIRD进程已停止" \
            -d "desp=⚠️ 警告：没有检测到 WAIRD 进程！\n\n时间：$(date '+%Y-%m-%d %H:%M:%S')\n主机：${HOSTNAME}\n目录：/home/hujiacong/zxd/Huawei/TransNet\n\n请及时检查！" \
            -o /dev/null
        
        echo "$(date): 已发送告警通知" >> $LOG_FILE
    fi
    
    # 等待5分钟
    sleep $INTERVAL
done