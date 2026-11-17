"""D2 验收：10 轮对话，第 2 轮问招生办电话，第 10 轮问"刚才那个电话是多少"。

用法：
    .venv\\Scripts\\python.exe verify_d2.py

改之前：第 10 轮答不上（历史只留最近 6 条，第 2 轮已被删）
改之后：应该能答出 0773-2290010
"""
import asyncio
import sqlite3
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import main

SID = "d2test"
TURNS = [
    "你好",
    "招生办的电话是多少？",      # ← 第 2 轮，答案要跨 8 轮活下来
    "图书馆能借几本书？",
    "普通专业学费多少钱？",
    "军训要多长时间？",
    "宿舍晚上几点关门？",
    "学校有几个校区？",
    "学费可以怎么交？",
    "新生报到是几号？",
    "刚才那个电话是多少？",       # ← 第 10 轮，考记忆
]


def reset(sid):
    with sqlite3.connect(main.DB_PATH) as conn:
        conn.execute("DELETE FROM messages WHERE session_id=?", (sid,))
        conn.execute("DELETE FROM summaries WHERE session_id=?", (sid,))
        conn.commit()


async def run():
    reset(SID)
    t0 = time.perf_counter()
    for i, q in enumerate(TURNS, 1):
        r = await main.chat(main.ChatRequest(query=q, session_id=SID))
        ans = r["answer"].replace("\n", " ")[:60]
        print(f"[{i:02d}] {q}  ->  {ans}")
    elapsed = time.perf_counter() - t0

    rows = main.load_all_messages(SID)
    summary = main.load_summary(SID)
    print("\n--- 结果 ---")
    print("总耗时 %.1fs（10 轮，含压缩）" % elapsed)
    print("messages 表剩 %d 条，摘要 %d 字" % (len(rows), len(summary)))
    print("摘要内容:", summary or "（无）")
    last = [c for _, role, c in rows if role == "assistant"][-1] if rows else ""
    ok = "0773" in last or "2290010" in last
    print("最后一问答对了吗:", "✅ 是" if ok else "❌ 否")


asyncio.run(run())
