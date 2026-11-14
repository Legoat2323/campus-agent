"""校园问答 holdout（验收集）——5 题，冻结，永不参与调参。

规则（和第一个项目一样）：
    1. 这 5 题不是开发集（test_qa_20.py 的 20 题），用来验证"没见过的题还准不准"
    2. 不出现在任何调参过程里；看了结果也不许回头改 prompt / 阈值
    3. 不自动判分——把回答打出来，人眼看，参考句只用来对照

为什么现在加：2026-11-14 用 20 题把阈值调到 1.05、把 prompt 也改过，
这 20 题已经"被调过"，分数会偏高。必须有一组没被碰过的题才能说真实水平。

用法（campus-agent 目录下）：
    .\\.venv\\Scripts\\python.exe test_qa_holdout.py
"""
import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

import main


# (问题, 参考句（原文摘要）, 有没有答案)
CASES = [
    (
        "招生办的电话是多少？",
        "0773-2290010（data/01_报到前.md『录取通知书什么时候到』第二条）",
        True,
    ),
    (
        "学费可以怎么交？",
        "网上迎新报到系统 http://yx.guet.edu.cn，或「桂林电子科技大学财务处」公众号"
        "（data/01_报到前.md『学费和住宿费多少？怎么交』末段）",
        True,
    ),
    (
        "军训期间要体检吗？需要空腹吗？",
        "军训时每个院会自行安排体检，需要空腹（data/03_入学第一周.md）",
        True,
    ),
    (
        "校医院在哪里？怎么报销？",
        "校医院在中央食堂附近，凭医保卡报销（data/04_校园生活.md）",
        True,
    ),
    (
        "宿舍里有洗衣机吗？",
        "无——文档只写了门禁时间与电器功率限制，没有洗衣机（data/04_校园生活.md）",
        False,
    ),
]


def run():
    print("=" * 88)
    print("holdout 5 题（冻结，人眼判）")
    print("=" * 88)

    for i, (question, ref, has_answer) in enumerate(CASES, start=1):
        context = main.get_school_context(question)
        try:
            result = asyncio.run(
                main.chat(main.ChatRequest(query=question, session_id=f"holdout-{i}"))
            )
            answer = str(result.get("answer", ""))
        except Exception as e:
            answer = f"[调用失败] {e}"

        print(f"\n[{i}/{5}] {question}")
        print(f"  参考：{ref}")
        print(f"  上下文：{len(context)} 字" + ("" if context.strip() else "（空 → 会走兜底）"))
        print(f"  回答：{answer}")
        print(f"  该有的行为：{'答出参考句里的关键信息' if has_answer else '明确说「没有相关资料」'}")
        print("  判定：______")

    print("\n" + "=" * 88)
    print("把 5 个判定填完，再算：答对 __/5")
    print("注意：这 5 题的分数才是对外能说的数字；20 题那套是开发集，会被调参抬高。")


if __name__ == "__main__":
    run()
