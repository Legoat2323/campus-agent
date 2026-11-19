"""用一个 MCP 客户端连上 mcp_server.py，验证工具能被发现和调用。"""
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])


async def main():
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            print(f"[握手] server = {info.serverInfo.name} v{info.serverInfo.version}")

            tools = await session.list_tools()
            print(f"[发现] 工具数 = {len(tools.tools)}")
            for t in tools.tools:
                print(f"   - {t.name}: {(t.description or '').splitlines()[0][:40]}")

            for name, args in [
                ("get_current_time", {}),
                ("search_school", {"question": "报到是几号"}),
                ("list_topics", {}),
            ]:
                res = await session.call_tool(name, args)
                text = res.content[0].text if res.content else "(空)"
                print(f"[调用] {name}{args} -> {text[:70].replace(chr(10), ' ')}")


if __name__ == "__main__":
    asyncio.run(main())
