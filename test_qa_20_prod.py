"""校园问答 v2 验收（走生产路径）——同一套 20 题，只换检索与 prompt。

和 test_qa_20.py 的唯一区别，也是最关键的一处：
    test_qa_20.py  → main.vectorstore.similarity_search(q, k=2) + 自己调模型 ← 不过阈值、无兜底
    本脚本          → main.chat(ChatRequest(...))                            ← 生产入口本身
                     （内部：get_school_context 带阈值 k=3 → 带 tools 的首轮 →
                      无资料且未调工具时返回 NO_DATA_ANSWER）

注意 v3 的修正：v2 用 get_school_context + build_messages 自己拼了一次调用，
绕过了 main.chat() 里"无资料且未调工具 → 兜底"那段（main.py:244-247），
于是把"空上下文时模型会编"当成了产品缺陷。实际上那是脚本缺陷。

为什么要跑这一版：
    上一版的「A 12/12、B 8/8」是在**没有阈值**的路径上测出来的。
    线上有阈值（main.py:61、main.py:151-156），B 组那些题很可能被整段滤光——
    那时拒答的功劳属于阈值，不属于 prompt。只有把两条路分开量，才知道该修哪一层。

用法（在 campus-agent 目录下，和 test_qa_20.py 放在一起）：
    .\\.venv\\Scripts\\python.exe test_qa_20_prod.py
"""
import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

import sqlite3
import main

try:
    from test_qa_20 import CASES, hit_any, says_no_answer
except ImportError:
    sys.exit("找不到 test_qa_20.py —— 请把本脚本和 test_qa_20.py 放在同一目录（campus-agent 根目录）。")


def ask_prod(question: str, session_id: str):
    """走真正的线上入口 main.chat()，返回 (context, answer)。

    context 单独取一次，只用于统计"阈值放行了几题"，不参与生成。
    """
    context = main.get_school_context(question)            # 带阈值过滤，k=3
    try:
        result = asyncio.run(
            main.chat(main.ChatRequest(query=question, session_id=session_id))
        )
        return context, str(result.get("answer", ""))
    except Exception as e:
        return context, f"[调用失败] {e}"


def run():
    a_total = a_ctx_full = a_ret = a_ans = 0
    b_total = b_ctx_full = b_ok = 0
    rows = []

    for i, (question, keywords, no_answer, note) in enumerate(CASES):
        # 每题一个独立 session_id，避免历史串扰。
        # ⚠️ 还要先清掉这个 sid 上一次跑留下的 messages/summaries，
        #    否则第二次跑时"摘要"会让 chat() 绕过多资料兜底，拒答率凭空掉。
        sid = f"prodtest-{i}"
        with sqlite3.connect(main.DB_PATH) as c:
            c.execute("DELETE FROM messages WHERE session_id=?", (sid,))
            c.execute("DELETE FROM summaries WHERE session_id=?", (sid,))
            c.commit()
        ctx, answer = ask_prod(question, sid)
        has_ctx = bool(ctx.strip())
        ctx_flag = f"{len(ctx)}字" if has_ctx else "空"
        flat = answer.replace("\n", " ")[:44]

        if no_answer:
            b_total += 1
            b_ctx_full += has_ctx
            ok = says_no_answer(answer)
            b_ok += ok
            rows.append((question, ctx_flag, "—", "✅" if ok else "❌", flat))
        else:
            a_total += 1
            a_ctx_full += has_ctx
            r_ok = has_ctx and hit_any(ctx, keywords)
            a_ok = hit_any(answer, keywords)
            a_ret += r_ok
            a_ans += a_ok
            rows.append((question, ctx_flag, "✅" if r_ok else "❌", "✅" if a_ok else "❌", flat))

    print("=" * 104)
    print(f"{'问题':<26}{'上下文':<8}{'检索':<6}{'回答':<6}实际答案（截断）")
    print("=" * 104)
    for q, c, r, a, ans in rows:
        # 中文宽度对齐不追求完美，能看就行
        print(f"{q:<22}{c:<8}{r:<5}{a:<5}{ans}")

    print("=" * 104)
    print("【A 组 12 题：资料里有答案】")
    print(f"  阈值放行（上下文非空）：{a_ctx_full}/{a_total}")
    print(f"  检索命中：{a_ret}/{a_total}    回答命中：{a_ans}/{a_total}")
    print("【B 组 8 题：资料里没有答案】")
    print(f"  阈值没拦住（上下文非空，只能靠 prompt 兜）：{b_ctx_full}/{b_total}")
    print(f"  拒答正确：{b_ok}/{b_total}")
    print("=" * 104)
    print("怎么读：")
    print("  A 组「阈值放行」< 12  → 阈值在误杀有答案的题（阈值定太高）")
    print("  B 组「阈值没拦住」> 0 → 这些题靠的是 prompt，不是阈值")
    print("  B 组「阈值没拦住」= 0 → 拒答是阈值的功劳，别记在 prompt 上")
    print("\n对照 test_qa_20.py（不过阈值、那句 prompt 也在）：A 组 12/12、B 组 8/8。")


if __name__ == "__main__":
    run()
