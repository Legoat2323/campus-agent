"""阈值标定探针：把 20 题的真实检索距离量出来，看 1.0 这条线该画在哪。

只读，不写库、不调生成模型。唯一的外部调用是 embedding（相似度检索本身）。

用法（campus-agent 目录下）：
    .\\.venv\\Scripts\\python.exe test_distance_probe.py

输出三部分：
    1. 每题 top-5 距离，以及「含答案片段」排第几、距离多少
    2. A 组（有答案）与 B 组（无答案）的距离分布对比
    3. 阈值扫描表：每个候选阈值下，A 组救回几题、B 组放行几题
"""
import sys

from dotenv import load_dotenv

load_dotenv()

import main

try:
    from test_qa_20 import CASES, hit_any
except ImportError:
    sys.exit("找不到 test_qa_20.py —— 请把本脚本和它放在同一目录（campus-agent 根目录）。")

TOP_K = 5
CURRENT = main.MAX_DISTANCE  # 现在线上用的线，1.0


def probe():
    a_rows = []  # (问题, 含答案片段距离, 名次, top1距离)
    b_rows = []  # (问题, top1距离)

    print("=" * 92)
    print(f"{'问题':<24}{'top1':>8}{'top2':>8}{'top3':>8}{'top4':>8}{'top5':>8}   含答案片段")
    print("=" * 92)

    for question, keywords, no_answer, note in CASES:
        results = main.vectorstore.similarity_search_with_score(question, k=TOP_K)
        dists = [score for _, score in results]

        if no_answer:
            b_rows.append((question, dists[0]))
            tag = "B组·无答案（命中即误放行）"
        else:
            rank, hit_dist = None, None
            for i, (doc, score) in enumerate(results, start=1):
                if hit_any(doc.page_content, keywords):
                    rank, hit_dist = i, score
                    break
            a_rows.append((question, hit_dist, rank, dists[0]))
            tag = f"第{rank}位 距离{hit_dist:.3f}" if rank else "top5 内没有"

        shown = "".join(f"{d:8.3f}" for d in dists)
        print(f"{question:<24}{shown}   {tag}")

    # ---------------- 分布 ----------------
    print("=" * 92)
    found = [r for r in a_rows if r[1] is not None]
    missing = [r for r in a_rows if r[1] is None]
    print(f"A 组 {len(a_rows)} 题：top5 内找到含答案片段 {len(found)} 题，没找到 {len(missing)} 题")
    if missing:
        print("  没找到（这是召回问题，调阈值救不了）：" + "、".join(r[0] for r in missing))
    if found:
        ds = sorted(r[1] for r in found)
        print(f"  含答案片段距离：最小 {ds[0]:.3f} / 中位 {ds[len(ds)//2]:.3f} / 最大 {ds[-1]:.3f}")
        over = [r for r in found if r[1] >= CURRENT]
        print(f"  当前线 {CURRENT} 会误杀 {len(over)} 题：" + "、".join(r[0] for r in over) if over else "  当前线没有误杀")

    b_ds = sorted(r[1] for r in b_rows)
    print(f"B 组 {len(b_rows)} 题：top1 距离 最小 {b_ds[0]:.3f} / 中位 {b_ds[len(b_ds)//2]:.3f} / 最大 {b_ds[-1]:.3f}")

    if found and b_rows:
        worst_a = max(r[1] for r in found)   # 最"不像"的有答案片段
        best_b = min(r[1] for r in b_rows)   # 最"像"的无答案题
        print("-" * 92)
        print(f"分界线参考：A 组最差 {worst_a:.3f}  ←→  B 组最好 {best_b:.3f}")
        if worst_a < best_b:
            print(f"  有空档，建议取值 {(worst_a + best_b) / 2:.2f}（区间 {worst_a:.2f}–{best_b:.2f} 都安全）")
        else:
            print("  两个分布重叠 → 单靠距离分不开，需要别的办法（提高 k、换 embedding、加 rerank）")

    # ---------------- 扫描 ----------------
    print("=" * 92)
    print(f"{'阈值':>6}{'A 救回':>10}{'A 误杀':>10}{'B 误放行':>12}")
    for t in (0.9, 1.0, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5):
        a_pass = sum(1 for r in found if r[1] < t)
        b_pass = sum(1 for _, d in b_rows if d < t)
        print(f"{t:>6.2f}{a_pass:>10}{len(found) - a_pass:>10}{b_pass:>12}")
    print("=" * 92)
    print("怎么读：A 救回 越多越好，B 误放行 越少越好；两列交叉点就是最佳线。")


if __name__ == "__main__":
    probe()
