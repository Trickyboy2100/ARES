# PG-JY Project Instructions

**同时只保持一个 Isaac Sim 窗口。启动新场景前必须先 kill 旧进程。**

```bash
# 关闭所有 Isaac Sim 实例
pkill -f "isaacsim/kit/kit" 2>/dev/null; sleep 2
# 若进程未退出，强制结束
ps aux | grep "isaacsim/kit/kit" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
```
