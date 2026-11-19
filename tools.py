"""工具定义（唯一来源）。campus 的 loop 和 MCP server 都从这里取。"""
import datetime
import glob
import os


from weather import get_weather


# ---------- 实现 ----------

def tool_get_weather(args: dict) -> str:
    city = (args or {}).get("city", "").strip()
    if not city:
        return "缺少 city 参数"
    try:
        return get_weather(city)
    except Exception as e:
        return f"查询天气失败：{e}"


def tool_search_school(args: dict) -> str:
    question = (args or {}).get("question", "").strip()
    if not question:
        return "缺少 question 参数"
    context = get_school_context(question)
    return context if context.strip() else "知识库里没有找到相关内容"


def tool_get_current_time(args: dict) -> str:
    now = datetime.datetime.now()
    weekday = "一二三四五六日"[now.weekday()]
    return f"{now:%Y-%m-%d %H:%M}，星期{weekday}"


def tool_list_topics(args: dict) -> str:
    """列出知识库覆盖的所有主题（读 data/*.md 的 ## 标题）。"""
    lines = []
    for path in sorted(glob.glob("data/*.md")):
        name = os.path.basename(path)
        for raw in open(path, encoding="utf-8"):
            if raw.startswith("## "):
                lines.append(f"{name}｜{raw[3:].strip()}")
    if not lines:
        return "知识库为空"
    return f"共 {len(lines)} 条，例如：\n" + "\n".join(lines[:20])


# ---------- 给模型看的 schema ----------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "用户要查询天气的时候用，其他情况别用",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "中文城市名 如 桂林 广州 深圳"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_school",
            "description": "用户要查询学校相关内容的时候用，其他情况别用如（问天气和时间的时候别用）",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "用户的原问题，照抄过去，不要自己改写或概括"}
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当用户问今天星期几，几点 几号查询时间的时候用其他情况不调用",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_topics",
            "description": "列出知识库当前覆盖的全部主题清单。用户问“你能回答哪些问题”“资料里都有什么”“有没有关于 XX 的内容”时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

TOOL_IMPL = {
    "get_weather": tool_get_weather,
    "search_school": tool_search_school,
    "get_current_time": tool_get_current_time,
    "list_topics": tool_list_topics,
}


if __name__ == "__main__":
    import json
    print(json.dumps(TOOL_SCHEMAS, ensure_ascii=False, indent=2))
    print("工具数：", len(TOOL_SCHEMAS), "实现数：", len(TOOL_IMPL))








