# -*- coding: utf-8 -*-
"""verify_d4.py —— 指代消解验证。

同一个会话两轮：
  第 1 轮：招生办的电话是多少？      （建立历史，答案里会出现 0773-2290010）
  第 2 轮：刚才那个电话是多少？      （这句话里没有"招生办"，检索抓不到）

跑两遍对比：关掉改写 vs 打开改写。
"""
import asyncio
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")

import main

SID = "d4test"
Q1 = "招生办的电话是多少？"
Q2 = "刚才那个电话是多少？"


def reset():
    with sqlite3.connect(main.DB_PATH) as c:
        c.execute("DELETE FROM messages WHERE session_id=?", (SID,))
        c.execute("DELETE FROM summaries WHERE session_id=?", (SID,))
        c.commit()


async def run_case(label, enable):
    main.ENABLE_QUERY_REWRITE = enable
    reset()
    print(f"\n===== {label}（ENABLE_QUERY_REWRITE={enable}）=====")

    r1 = await main.chat(main.ChatRequest(query=Q1, session_id=SID))
    print(f"[1] {Q1} -> {r1['answer'][:60]}")

    search_q = main.rewrite_query(Q2, SID) if enable else Q2
    ctx = main.get_school_context(search_q)
    hit = ("0773" in ctx) or ("2290010" in ctx)
    print(f"[2] 检索用的句子：{search_q}")
    print(f"[2] 是否检索到那一段：{'✅' if hit else '❌'}（context {len(ctx)} 字）")

    r2 = await main.chat(main.ChatRequest(query=Q2, session_id=SID))
    ok = ("0773" in r2["answer"]) or ("2290010" in r2["answer"])
    print(f"[2] {Q2} -> {r2['answer'][:60]}")
    print(f"[2] 答对了吗：{'✅' if ok else '❌'}")
    return hit, ok


async def amain():
    off = await run_case("对照组：不改写", False)
    on = await run_case("实验组：带历史改写", True)
    print("\n===== 结果 =====")
    print(f"关闭改写：检索 {'✅' if off[0] else '❌'}  回答 {'✅' if off[1] else '❌'}")
    print(f"打开改写：检索 {'✅' if on[0] else '❌'}  回答 {'✅' if on[1] else '❌'}")
    reset()


asyncio.run(amain())
