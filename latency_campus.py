"""校园 Agent 延迟测试：8 题 × 3 次，热身 1 次丢弃。

指标：平均 / P50 / P95 / 最慢 / 最快。
用法：.\.venv\Scripts\python.exe latency_campus.py
"""
import asyncio
import statistics
import time

import main

QUESTIONS = [
    "新生报到是几号？",                 # 有答案（知识库）
    "宿舍晚上几点关门？",               # 有答案
    "学费多少钱？",                     # 有答案
    "校医院在哪儿？",                   # 有答案
    "深圳今天的天气怎么样？",           # 走工具
    "图书馆几点开门？",                 # 无答案（应拒答）
    "宿舍有空调吗？",                   # 无答案（应拒答）
    "转专业有名额限制吗？",             # 无答案（应拒答）
]
TIMES = 3


def one(question: str, session_id: str) -> float:
    t0 = time.perf_counter()
    try:
        asyncio.run(main.chat(main.ChatRequest(query=question, session_id=session_id)))
    except Exception as e:                      # 单次失败不计入，但要打印
        print(f"  [失败] {question[:12]} -> {type(e).__name__}")
        return -1.0
    return time.perf_counter() - t0


def main_run():
    print("热身（不计入）...")
    one(QUESTIONS[0], "warmup")
    samples = []
    for i, q in enumerate(QUESTIONS):
        for r in range(TIMES):
            dt = one(q, f"lat-{i}-{r}")          # 每题独立 session，防记忆串扰
            if dt > 0:
                samples.append(dt)
                print(f"  [{i+1}-{r+1}] {q[:14]:<16} {dt:.2f}s")

    samples.sort()
    n = len(samples)
    p50 = statistics.median(samples)
    p95 = samples[min(n - 1, int(round(n * 0.95)) - 1)]
    print("\n--- 结果 ---")
    print(f"样本 {n}  平均 {sum(samples)/n:.2f}s  P50 {p50:.2f}s  "
          f"P95 {p95:.2f}s  最慢 {samples[-1]:.2f}s  最快 {samples[0]:.2f}s")


if __name__ == "__main__":
    main_run()
