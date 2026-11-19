"""把 campus 的工具包成 MCP server（stdio 传输）。

设计要点：工具定义只在 tools.py 里写一份，这里只是"换个方式挂出去"。
MCP 的 schema 由 FastMCP 从函数签名 + docstring 自动生成，
所以每个工具的 docstring 就是模型看到的 description——要写清"什么时候用"。
"""
from mcp.server.fastmcp import FastMCP

# 预热：Chroma 必须在事件循环启动前初始化。
# 如果等到工具被调用时才 import main，chromadb 的 Rust 客户端
# 会在运行中的事件循环里创建，导致整个 server 卡死（实测卡 150s 超时）。
import main  # noqa: F401  仅为初始化向量库
from tools import TOOL_IMPL

mcp = FastMCP("campus-tools")


@mcp.tool()
def get_weather(city: str) -> str:
    """查询指定城市的实时天气（温度、天气状况、风速）。
    用户问某地天气、气温、是否下雨时使用。问校园规定、时间时不要用。"""
    return TOOL_IMPL["get_weather"]({"city": city})


@mcp.tool()
def search_school(question: str) -> str:
    """在桂电校园知识库中检索原文片段。
    用户询问报到时间、学费、宿舍门禁、军训安排、校区地址、招生电话等
    桂电相关具体事实时使用。查询天气或当前时间时不要使用。"""
    return TOOL_IMPL["search_school"]({"question": question})


@mcp.tool()
def get_current_time() -> str:
    """获取当前的日期、时间和星期。用户问现在几点、今天几号、星期几时使用。"""
    return TOOL_IMPL["get_current_time"]({})


@mcp.tool()
def list_topics() -> str:
    """列出知识库当前覆盖的全部主题清单。
    用户问"你能回答哪些问题""资料里都有什么"时使用。"""
    return TOOL_IMPL["list_topics"]({})


if __name__ == "__main__":
    mcp.run()
