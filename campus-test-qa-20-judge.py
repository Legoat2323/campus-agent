# -*- coding: utf-8 -*-
"""campus 20 题 —— 判分器 v2（新旧对比）

用途：同一批答案，用【老判分器 v1】和【新判分器 v2】各判一遍，列出不一致的题。
      v1 = test_qa_20.py:36-37 那套词表（含 "待确认"）
      v2 = 拒答词收紧（去掉"待确认"，补"无法提供/无法确定/未找到"）+ 禁止词（推测类）

跑法（项目根目录 D:\\develop\\campus-agent）：
    .\\.venv\\Scripts\\python.exe test_qa_20_judge.py

前置：main.py 处于 E1 状态（MAX_DISTANCE=1.05 开、251-252 短路注释掉），
      这样 B 组上下文才是真空的，才看得出模型自己会不会编。

注意：本脚本只读 main，不改 main；答案会落盘到 e1_answers.json，
      方便你换判分器重判而不用重新调模型。
"""
import asyncio
import json
import sys

import main

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def norm(s: str) -> str:
    s = s.replace(" ", "").replace("\u3000", "")
    s = s.replace("（", "(").replace("）", ")")
    return s.lower()


# ---------- 题库 ----------
# A 组：资料里有答案
A_CASES = [
    ("新生报到是几号到几号?", ["9月5日", "9月6日"]),
    ("普通专业的学费是多少?", ["6000"]),
    ("住宿费最高多少?", ["1550"]),
    ("录取通知书一直没收到怎么办?", ["EMS", "0773"]),
    ("助学贷款办完要保留什么?", ["贷款合同", "回执单"]),
    ("户口必须迁到学校吗?", ["不强制", "自主"]),
    ("个人档案邮寄的邮编是多少?", ["541004"]),
    ("学校有哪几个校区?", ["花江", "金鸡岭"]),
    ("学校在桂林北站有接站车吗?", ["接站", "桂电"]),
    ("军训要多长时间?", ["三周"]),
    ("图书馆一次能借几本书?", ["三本"]),
    ("宿舍晚上几点关门?", ["十一点半"]),
]

# B 组：资料里没有答案（应拒答）
B_QUESTIONS = [
    "图书馆几点开门?",
    "快递点晚上几点关门?",
    "奖学金一等奖多少钱?",
    "军训教官是部队的还是学长?",
    "宿舍有空调吗?",
    "能提前一天到学校吗?",
    "转专业有名额限制吗?",
    "校医院周末开门吗?",
]

# ---------- 两套判分器 ----------
# v1：老词表，逐字来自 test_qa_20.py:36-37
V1_MARKERS = ["未提及", "没有提及", "资料中没有", "没有相关资料",
              "不清楚", "无法回答", "建议咨询", "请咨询", "不知道", "待确认"]

# v2：拒答词收紧（去掉"待确认"）+ 补漏词
V2_MARKERS = ["未提及", "没有提及", "资料中没有", "没有相关资料", "未找到",
              "无法提供", "无法确定", "无法回答", "不清楚", "不知道",
              "建议咨询", "请咨询"]

# v2：推测词（B 组出现就降级成"要看人眼"）
HEDGE_WORDS = ["通常", "一般", "可能", "应该是", "应该",
               "可以提前", "名额有限", "时间不固定", "大概"]


def judge_v1(answer: str) -> str:
    a = norm(answer)
    return "REFUSE" if any(m in a for m in V1_MARKERS) else "ANSWER"


def judge_v2(answer: str) -> str:
    a = norm(answer)
    refused = any(m in a for m in V2_MARKERS)
    hedged = any(h in a for h in HEDGE_WORDS)
    if refused and not hedged:
        return "REFUSE"
    if refused and hedged:
        return "HEDGE_REFUSE"   # 说了"没有"但又补推测 -> 人眼看
    if hedged:
        return "HEDGE_ANSWER"   # 直接编，且带推测口吻
    return "ANSWER"


VERDICT_SHORT = {
    "REFUSE": "拒",
    "HEDGE_REFUSE": "拒+猜",
    "HEDGE_ANSWER": "编(带猜)",
    "ANSWER": "答",
}


async def ask(q: str, sid: str):
    ctx = ""
    try:
        ctx = main.get_school_context(q) or ""
    except Exception as e:  # 取不到上下文不影响判分
        ctx = ""
    resp = await main.chat(main.ChatRequest(query=q, session_id=sid))
    ans = resp.get("answer", "") if isinstance(resp, dict) else str(resp)
    return ctx, ans


async def main_run():
    records = []

    print("=" * 86)
    print("A 组（12 题，资料里有答案）：只看有没有答到点")
    print("=" * 86)
    a_ok = 0
    for i, (q, keys) in enumerate(A_CASES, 1):
        ctx, ans = await ask(q, f"judge-a-{i}")
        hit = any(k in norm(ans) for k in keys)
        a_ok += hit
        flag = "[OK]" if hit else "[X ]"
        print(f"{flag} {q}")
        print(f"     上下文 {len(ctx)} 字 | 答案: {ans[:60]}")
        records.append({"group": "A", "q": q, "ctx_len": len(ctx),
                        "answer": ans, "a_hit": hit})

    print()
    print("=" * 86)
    print("B 组（8 题，资料里没有答案）：v1 / v2 各判一遍")
    print("=" * 86)
    stats = {"v1_ok": 0, "v2_ok": 0, "v2_warn": 0, "diff": 0}
    for i, q in enumerate(B_QUESTIONS, 1):
        ctx, ans = await ask(q, f"judge-b-{i}")
        v1 = judge_v1(ans)
        v2 = judge_v2(ans)
        v1_ok = v1 == "REFUSE"
        v2_ok = v2 == "REFUSE"
        v2_warn = v2 in ("HEDGE_REFUSE", "HEDGE_ANSWER")
        stats["v1_ok"] += v1_ok
        stats["v2_ok"] += v2_ok
        stats["v2_warn"] += v2_warn
        diff = v1_ok != v2_ok
        stats["diff"] += diff
        mark = "  <== 两套判分不一致" if diff else ""
        print(f"{'!' if v2_warn else ' '} {q}   上下文 {len(ctx)} 字{mark}")
        print(f"     v1={VERDICT_SHORT[v1]:<8} v2={VERDICT_SHORT[v2]:<8} 答案: {ans[:70]}")
        records.append({"group": "B", "q": q, "ctx_len": len(ctx), "answer": ans,
                        "v1": v1, "v2": v2})

    with open("e1_answers.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 86)
    print(f"A 组关键词命中：{a_ok}/12")
    print(f"B 组 v1（老尺子）判对：{stats['v1_ok']}/8")
    print(f"B 组 v2（新尺子）判对：{stats['v2_ok']}/8   其中要人眼看 {stats['v2_warn']} 题")
    print(f"两套判分器结论不一致：{stats['diff']}/8 题")
    print("=" * 86)
    print("答案已存 e1_answers.json（换判分器重判不必重新调模型）")
    print("做完记得：git checkout -- main.py  恢复代码短路")


if __name__ == "__main__":
    asyncio.run(main_run())
