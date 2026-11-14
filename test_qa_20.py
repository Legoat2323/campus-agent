"""校园问答效果测试 v2：20 个问题（12 有答案 + 8 知识库里没有）。

放到 campus-agent 目录下运行（和 main.py 同级）：
    .\\.venv\\Scripts\\python.exe test_qa_20.py

和旧版 test_qa.py 的区别：
1. 8 题 → 20 题。8 题里错 1 个就是 87.5%，说明不了问题。
2. **加 8 道「知识库里没有」的题**：都是新生真会问、但资料没写的。
   → 这 8 题才是重点：模型会不会拿相邻的内容硬编。
3. 判分改成「多个关键词命中任一」，并去掉空格再比
   （旧版单个关键词，`9月5` 遇到 `9 月 5` 就判错——那是尺子的问题，不是系统的问题）。

指标分两组：
    A 组（12 题）：检索命中 + 回答命中，分母 12
    B 组（8 题）：拒答正确率（说「未提及/没有/不清楚」才算对，硬编就是错），分母 8

判读：
    检索 ❌ 回答 ✅ → 先怀疑关键词，把片段打出来看
    检索 ✅ 回答 ❌ → 资料到了没用上，是 prompt 问题
    B 组 ❌       → 模型在编，这类最该修
"""
import re

from dotenv import load_dotenv

load_dotenv()

import main  # 复用 main.py 里初始化好的 vectorstore 和 client


def norm(text: str) -> str:
    text = (text or "").replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", "", text)


NO_ANSWER_MARKERS = ["未提及", "没有提及", "资料中没有", "没有相关资料",
                     "不清楚", "无法回答", "建议咨询", "请咨询", "不知道", "待确认"]

# (问题, 关键词列表, 是否「知识库里没有」, 备注)
# A 组关键词全部从 data/*.md 原文里抄
CASES = [
    # ---------- A 组：资料里有答案（12 题） ----------
    ("新生报到是几号到几号？", ["9月5日至9月6日", "9月5"], False, "01 报到时间"),
    ("普通专业的学费是多少？", ["6000"], False, "01 学费"),
    ("住宿费最高多少？", ["1550"], False, "01 住宿费"),
    ("录取通知书一直没收到怎么办？", ["2290010", "EMS"], False, "01 通知书"),
    ("助学贷款办完要保留什么？", ["回执"], False, "01 助学贷款"),
    ("户口必须迁到学校吗？", ["不强制", "自主"], False, "01 户口"),
    ("个人档案邮寄的邮编是多少？", ["541004"], False, "01 档案"),
    ("学校有哪几个校区？", ["花江", "金鸡岭"], False, "02 校区"),
    ("学校在桂林北站有接站车吗？", ["桂林北站", "接待站"], False, "02 接站"),
    ("军训要多长时间？", ["三周"], False, "03 军训"),
    ("图书馆一次能借几本书？", ["三本"], False, "04 图书馆"),
    ("宿舍晚上几点关门？", ["十一点半"], False, "04 宿舍门禁"),

    # ---------- B 组：资料里没有（8 题） ----------
    # 这些题目都「贴着已有内容」，检索一定会捞到相邻片段 →
    # 模型是老实说没有，还是拿隔壁段的内容硬凑，就看这 8 题
    ("图书馆几点开门？", [], True, "04 只有刷脸+借书数，没有开放时间"),
    ("快递点晚上几点关门？", [], True, "04 只有位置和取件方式"),
    ("奖学金一等奖多少钱？", [], True, "03 只说去官网查指南"),
    ("军训教官是部队的还是学长？", [], True, "03 只写时长和服装"),
    ("宿舍有空调吗？", [], True, "04 只写门禁和电器功率"),
    ("能提前一天到学校吗？", [], True, "02 只写了延期报到"),
    ("转专业有名额限制吗？", [], True, "03 只写政策和时间点"),
    ("校医院周末开门吗？", [], True, "04 只写位置和报销"),
]


def hit_any(text: str, keywords) -> bool:
    t = norm(text)
    return any(norm(k) in t for k in keywords)


def says_no_answer(text: str) -> bool:
    return any(m in text for m in NO_ANSWER_MARKERS)


def run():
    a_total = a_ret = a_ans = 0
    b_total = b_ok = 0
    fails = []

    print("=" * 96)
    print(f"{'问题':<26}{'检索':<6}{'回答':<6}实际答案（截断 46 字）")
    print("=" * 96)

    for question, keywords, no_answer, note in CASES:
        try:
            docs = main.vectorstore.similarity_search(question, k=2)
            context = "\n".join(d.page_content for d in docs)
        except Exception as e:
            context, docs = "", []
            print(f"[检索失败] {e}")

        try:
            resp = main.client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system",
                     "content": "你是一个校园助手。请只根据以下资料回答；资料里没有的，"
                                "就直接说「资料中未提及」，不要推测、不要编：\n" + context},
                    {"role": "user", "content": question},
                ],
                temperature=0.1,
            )
            answer = resp.choices[0].message.content or ""
        except Exception as e:
            answer = f"[调用失败] {e}"

        if no_answer:
            b_total += 1
            ok = says_no_answer(answer)
            b_ok += ok
            print(f"{question:<26}{'—':<6}{'✅' if ok else '❌':<6}{answer.replace(chr(10), ' ')[:46]}")
            if not ok:
                fails.append((question, "该说没有却没说：" + note, answer))
        else:
            a_total += 1
            r_ok = hit_any(context, keywords)
            a_ok = hit_any(answer, keywords)
            a_ret += r_ok
            a_ans += a_ok
            print(f"{question:<26}{'✅' if r_ok else '❌':<6}{'✅' if a_ok else '❌':<6}{answer.replace(chr(10), ' ')[:46]}")
            if not r_ok:
                fails.append((question, "检索未命中：" + note, answer))

    print("=" * 96)
    print(f"A 组（有答案 {a_total} 题）检索命中率：{a_ret}/{a_total} = {a_ret / a_total:.0%}")
    print(f"A 组（有答案 {a_total} 题）回答命中率：{a_ans}/{a_total} = {a_ans / a_total:.0%}")
    print(f"B 组（无答案 {b_total} 题）拒答正确率：{b_ok}/{b_total} = {b_ok / b_total:.0%}")
    print("=" * 96)

    if fails:
        print("\n需要人眼看的题：")
        for q, why, a in fails:
            print(f"  - {q}  [{why}]\n    答：{a[:80].replace(chr(10), ' ')}")

    print("\n注意：样本 20 题，20/20 的 95% 置信区间约 83%–100%，别对外说「准确率 100%」。")


if __name__ == "__main__":
    run()
