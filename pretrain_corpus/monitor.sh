#!/bin/bash
echo "=== MEM/SWAP ==="
free -g
echo
echo "=== TOP WORKERS ==="
ps -eo pid,rss,%cpu,%mem,etime,cmd --sort=-rss | grep -E "download.py|python3" | grep -v grep | head -20
echo
echo "=== TOTALS ==="
ps -eo rss,cmd | grep "download.py\|python3" | grep -v grep | awk '{sum+=$1} END {printf "worker RSS total: %.1f GB\n", sum/1024/1024}'
