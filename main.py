"""校园智能 Agent 问答服务。

改造说明（相对上一版）：
1. 会话记忆从内存 dict 换成 SQLite —— 重启不丢，多进程也能共享
2. 错误处理改用 HTTPException —— 客户端能区分成功/失败，不再一律 200
3. print 换成 logging —— 能落盘、能分级、带时间戳
"""
import json
import re
import logging
import os
import sqlite3
import time
from contextlib import closing

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, RateLimitError
from pydantic import BaseModel

from weather import CITY_COORD, get_weather
from tools import TOOL_SCHEMAS as tools, TOOL_IMPL
from langchain_community.vectorstores import Chroma  #这是langchain的向量数据库
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ----------------------------------------------------------------------------
# 日志：按时间戳 + 级别输出，以后要落盘只需再加一个 FileHandler
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("campus")


# ----------------------------------------------------------------------------
# 模型与向量库
# ----------------------------------------------------------------------------
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

#向量化的工具
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    model="embedding-2",
    check_embedding_ctx_length=False,
)
log.info("正在连接向量数据库 ...")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

app = FastAPI(title="我的校园Agent")

# 检索距离阈值：距离 >= 它的片段视为不相关，直接丢弃。
# 2026-11-14 用 20 题重新标定（test_distance_probe.py）：
#   A 组 12 题「含答案片段」距离 0.555 ~ 1.162，B 组 8 题「最相似片段」1.024 ~ 1.203。
#   两个分布重叠 → 距离分不开「话题像不像」和「事实在不在」。
#   取 1.05：A 组放行 9/12、B 组 0 题被误放行（靠 prompt 兜底 8/8），综合 17/20。
#   放宽到 1.20 能救回 12/12，但 B 组会有一半题拿到「相近但不含答案」的上下文，飘到 5~8/8。  所以这里有个取舍的存在
MAX_DISTANCE = 1.05
MODEL_NAME = "glm-4-flash"
MODEL_TEMPERATURE = 0.0  # 0.0：有资料时结果可复现（实测两次均 11/12）；无资料靠代码短路兜底

# ----------------------------------------------------------------------------
# 会话记忆：SQLite
#
# 为什么不用全局 dict：
#   - 进程重启就丢
#   - uvicorn 开多个 worker 时各存各的
#   - 只有上限没有清理，内存只涨不降
# ----------------------------------------------------------------------------
DB_PATH = "memory.db"
HISTORY_LIMIT = 6  # 只保留最近 3 轮（user + assistant 各 3 条）


def init_db() -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        #用来记录用户和ai说的每一句对话 （短期记忆）
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id, id)"  #建索引
        )
        # D2：被"压缩掉"的历史不再直接删，而是先压成一段摘要存这里
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                session_id TEXT PRIMARY KEY,
                content    TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # D3：一次请求一行，events 是一个 JSON 数组（检索/每轮模型/工具/答案）   trace 全链路追踪
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                events     TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id, id)"
        )
        conn.commit()


def load_history(session_id: str, limit: int = HISTORY_LIMIT) -> list[dict]:
    """取最近 limit 条消息，按时间正序返回，可直接喂给模型。"""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": role, "content": text} for role, text in reversed(rows)]


def append_message(session_id: str, role: str, content: str) -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()   #数据库完成后必须提交事务 否则数据只存在内存里


def trim_history(session_id: str, keep: int = HISTORY_LIMIT) -> None:
    """只保留最近 keep 条，避免数据库无限增长。
    括号里的子查询是：选出 session_id 对应的最新 keep 个 消息的id
    外面删除的逻辑：把该用户除了这 keep 个消息以外的所有消息都删掉
    """
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            "DELETE FROM messages WHERE session_id=? AND id NOT IN "
            "(SELECT id FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?)",
            (session_id, session_id, keep),
        )
        conn.commit()


# ----------------------------------------------------------------------------
# D2：上下文压缩
#   原来 trim_history 直接把 6 条以外的对话删掉 —— 信息就真没了。
#   现在改成：超出的部分先让模型压成一段"事实清单"，存进 summaries，
#   再删原消息。下次对话时把摘要拼进 system prompt。
# ----------------------------------------------------------------------------
SUMMARY_TRIGGER = 8   # 消息数超过它才触发压缩
KEEP_RECENT = 6       # 最近 6 条（3 轮）保留原文
SUMMARY_MAX_CHARS = 200

SUMMARY_PROMPT = (
    "把下面的对话压缩成一段不超过 {limit} 字的事实清单。\n"
    "只保留：用户问过什么、回答里给过哪些关键数字/时间/地点/电话。\n"
    "不要寒暄，不要解释，不要评论，用'；'分隔。\n\n{dialog}"
)

# 只有当"旧摘要 + 新摘要"超过上限时才用这一次，把两段再压成一段
MERGE_PROMPT = (
    "下面是一份事实清单，请在不丢失任何事实的前提下压缩到 {limit} 字以内。\n"
    "数字、电话、时间、地点一律照抄，不许省略。用'；'分隔，不要解释。\n\n{facts}"
)


def load_summary(session_id: str) -> str:
    """取这个会话已有的摘要，没有就返回空串。"""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute(
            "SELECT content FROM summaries WHERE session_id=?", (session_id,)
        ).fetchone()
    return row[0] if row else ""


def save_summary(session_id: str, content: str) -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            "INSERT INTO summaries (session_id, content, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(session_id) DO UPDATE SET content=excluded.content, updated_at=CURRENT_TIMESTAMP",
            (session_id, content),
        )
        conn.commit()


def load_all_messages(session_id: str) -> list[tuple[int, str, str]]:
    """取这个会话的全部消息（id, role, content），按时间正序。压缩要用 id 才能删准。"""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT id, role, content FROM messages WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
    return rows


def delete_messages(ids: list[int]) -> None:
    if not ids:
        return
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.executemany("DELETE FROM messages WHERE id=?", [(i,) for i in ids])
        conn.commit()


def save_trace(session_id: str, events: list[dict]) -> None:
    """把一次请求的事件列表存下来。失败不影响响应（trace 是辅助，不是主流程）。"""
    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            conn.execute(
                "INSERT INTO traces (session_id, events) VALUES (?, ?)",
                (session_id, json.dumps(events, ensure_ascii=False)),
            )
            conn.commit()
    except Exception as e:
        log.warning("trace 落库失败 session=%s err=%s", session_id, e)


def get_traces(session_id: str, limit: int = 20) -> list[dict]:
    """取某个 session 最近的 trace，按时间倒序（最新的在前）。"""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT id, created_at, events FROM traces WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    out = []
    for tid, created_at, events in rows:
        try:
            parsed = json.loads(events)
        except json.JSONDecodeError:
            parsed = []
        out.append({"trace_id": tid, "created_at": created_at, "events": parsed})
    return out


def maybe_compress(session_id: str) -> None:
    """消息超过阈值时，把最早那批压成摘要再删。

    返回 None；失败不要抛异常（压缩失败就让消息留着，下次再试）。
    """
    old_summary = load_summary(session_id)  # ⚠️ 不能忽略


    rows = load_all_messages(session_id)
    if len(rows) <= SUMMARY_TRIGGER:
        return

    old_rows = rows[:-KEEP_RECENT]
    if not old_rows:
        return


    dialog_parts = []
    for _, role, content in old_rows:
        who = "用户" if role == "user" else "助手"
        dialog_parts.append(f"{who}: {content}")
    dialog = "\n".join(dialog_parts)

    def _ask(prompt: str) -> str:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return (r.choices[0].message.content or "").strip()

    try:
        # 只让它压"新增的那几条"。合并旧摘要这件事交给代码做 —— 模型做合并会丢事实。
        piece = _ask(SUMMARY_PROMPT.format(limit=SUMMARY_MAX_CHARS, dialog=dialog))
        if not piece:
            log.warning("压缩返回空 session=%s", session_id)
            return

        merged = "；".join(x for x in (old_summary, piece) if x)

        # 拼接后超长，才让模型统一再压一次（此时要求"数字照抄、不许省略"）
        if len(merged) > SUMMARY_MAX_CHARS:
            merged = _ask(MERGE_PROMPT.format(limit=SUMMARY_MAX_CHARS, facts=merged)) or merged

        save_summary(session_id, merged[: SUMMARY_MAX_CHARS * 2])
        delete_messages([mid for mid, _, _ in old_rows])
        log.info("压缩完成 session=%s 压掉 %d 条 -> 摘要 %d 字",
                 session_id, len(old_rows), len(merged))
    except Exception as e:
        log.warning("压缩失败 session=%s err=%s", session_id, e)

init_db()


class ChatRequest(BaseModel):
    session_id: str = "default_user"
    query: str


# ----------------------------------------------------------------------------
# 检索
# ----------------------------------------------------------------------------
SYSTEM_PROMPT_TMPL = """你是桂林电子科技大学的校园助手，服务对象是新入学的大一新生。

【必须遵守的规则】
1. 只依据下面【资料】里的内容回答，不要使用你自己的知识。
2. 如果【资料】里没有答案，直接回复："这个问题我这边没有相关资料，建议咨询辅导员或查看学校官网。"
3. 禁止编造店名、价格、电话、时间等具体信息。
4. 回答简洁，不超过 5 句话。
5. 如果【资料】里没有内容，但用户问的是天气这类可以用工具查询的信息，请调用工具，不要直接拒答。

【资料】
{context}"""

NO_DATA_ANSWER = "这个问题我这边没有相关资料，建议咨询辅导员或查看学校官网。"

# D4：带历史的查询改写（指代消解）
# 为什么需要：用户第二句常是"刚才那个电话是多少"，这句话里没有"招生办"，
# 向量和 BM25 都抓不到；改写把它补全成"招生办的电话是多少"再去检索。
ENABLE_QUERY_REWRITE = True
REWRITE_PROMPT = """把【当前问题】改写成一句独立、可直接用于检索的问句。

规则：
1. 如果当前问题里有指代词（刚才 / 上面 / 这个 / 那个 / 它 / 他 / 该 / 此），必须从【对话历史】里找到它指的具体事物，把那个词**替换进去**。
2. 改写后的句子里**不许再出现任何指代词**。出现就说明你没做完。
3. 如果当前问题本身已经完整（没有指代），**原样返回**，不许加词、不许扩写、不许换说法。
4. **不许添加原文没有的词**——尤其时间词（今天 / 现在 / 最近 / 最新）和限定词（2026级 / 全部 / 详细）。
   加词会改变向量方向，让原本能命中的片段掉出阈值。
5. 只输出改写后的那一句，不要解释，不要加引号。
6. 校园/教务场景下的同音错别字，按最可能的正确词纠正：
   报道→报到、帐号→账号、宿社→宿舍、图书管→图书馆。
   只改这些明显的同音字，不要改其他任何字。

示例：
历史：
用户: 招生办的电话是多少？
助手: 招生办的电话是0773-2290010。
当前问题：刚才那个电话是多少？
正确改写：招生办的电话是多少？
错误改写：刚才的电话是多少？（还留着"刚才"，等于没改）

【对话历史】
{history}

【当前问题】
{query}"""


def get_school_context(question: str, fallback: str | None = None) -> str:
    """检索知识库，只返回相关度够高的片段。

    fallback：查询改写后如果一条都没命中，就用用户原句再查一次。      fallback是原问句
    改写的本意是帮忙，但加词会让向量偏移、把能命中的片段推出阈值——
    这时候退回原句比"诚实地答不知道"更好。
    """
    try:
        results = vectorstore.similarity_search_with_score(question, k=3)
        useful = [doc for doc, score in results if score < MAX_DISTANCE]
        log.info("检索命中 %d/%d 条（阈值 %.1f）", len(useful), len(results), MAX_DISTANCE)

        if not useful and fallback and fallback != question:
            results2 = vectorstore.similarity_search_with_score(fallback, k=3)
            useful = [doc for doc, score in results2 if score < MAX_DISTANCE]
            log.info("改写后 0 命中，回退原句「%s」命中 %d/%d 条", fallback, len(useful), len(results2))

        return "\n".join(doc.page_content for doc in useful)
    except Exception:
        log.exception("检索出错，按无资料处理")
        return ""


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
# 天气工具的真实实现在 weather.py：
#   - 数据来自 Open-Meteo（免费、无需 API key）
#   - 内置常用城市别名表，解决中文地名匹配不准的问题
#   - 带 10 分钟缓存，避免重复请求




# ----------------------------------------------------------------------------
# Agent 循环（D1）：工具注册表 + 多轮调用 + 终止条件
# ----------------------------------------------------------------------------

MAX_TOOL_ROUNDS = 3


# ---- 天气类问题：代码直接答，不交给模型判断"要不要调工具" ----
# 理由：模型有时不调工具、直接凭常识编天气（实测出现过"10℃到20℃之间"这种不存在的区间）。
# 天气有确定性数据源，用代码兜住，模型只负责转述 → 从根上不可能编。
_WEATHER_RE = re.compile(
    r"天气|气温|温度|下雨|下雪|冷不冷|热不热|带伞|雨伞|晒不晒|防晒|紫外线|潮不潮|闷不闷"
)


def recent_city(session_id: str, lookback: int = 8) -> str | None:
    """从最近几轮对话里找城市名（越近越优先）。

    反例：上一轮聊"广州天气"，这一轮只问"下雨了吗"——
    当前句没有城市，但意图继承了上一轮的广州。
    """
    for _id, _role, content in reversed(load_all_messages(session_id)[-lookback:]):
        c = content or ""
        for city in CITY_COORD:
            if city in c:
                return city
    return None


def try_answer_weather(query: str, force: bool = False, session_id: str | None = None) -> str | None:
    """命中天气问题就返回答案（或固定失败话术）；不是天气问题返回 None。

    force=True 用于"上下文已判定是天气追问"的场景（用户只回了一个城市名，
    句子里没有"天气"两个字，但意图明确）。
    当前句没城市时，先回退到最近几轮说过的城市；还没有才反问。
    """
    if not force and not _WEATHER_RE.search(query):
        return None
    city = next((c for c in CITY_COORD if c in query), None)
    if city is None and session_id:
        city = recent_city(session_id)
        if city:
            log.info("天气问题未带城市，沿用上文城市=%s session=%s", city, session_id)
    if city is None:
        return "你想查哪个城市的天气？告诉我城市名就行。"
    return get_weather(city)          # 失败时它自己返回"天气服务暂时连不上…"


def wants_weather(query: str, session_id: str) -> bool:
    """判断"这一轮到底是不是在问天气"——不能只看当前句子。

    反例：上一轮助手问"你想查哪个城市的天气？"，用户只回"桂林"，
    当前句子里没有"天气"两个字，但意图明显是天气。
    """
    if _WEATHER_RE.search(query):
        return True
    city = next((c for c in CITY_COORD if c in query), None)
    if city is None:
        return False
    # 这一轮只有一个城市名（如"桂林"）：看上一轮助手是不是在聊天气
    for _id, role, content in reversed(load_all_messages(session_id)[-4:]):
        c = content or ""
        if role == "assistant" and ("℃" in c or "天气" in c):
            return True
    return False


# ---- 工具调用缓存：同一工具 + 同样参数，TTL 内直接复用，避免重复外部请求 ----
_TOOL_CACHE: dict[tuple[str, str], tuple[float, str]] = {}
_TOOL_CACHE_TTL = 300          # 秒；天气这类时效性数据 5 分钟足够


def call_tool(name: str, args: dict, trace: list | None = None) -> str:
    """带 TTL 缓存的工具调用。命中缓存时不发起外部请求。"""
    key = (name, json.dumps(args, ensure_ascii=False, sort_keys=True))
    now = time.time()
    hit = _TOOL_CACHE.get(key)
    if hit and now - hit[0] < _TOOL_CACHE_TTL:
        log.info("工具缓存命中 tool=%s args=%s", name, key[1])
        if trace is not None:
            trace.append({"kind": "cache", "name": name, "args": args})
        return hit[1]

    handler = TOOL_IMPL.get(name)
    if handler is None:
        return f"未知工具：{name}"
    try:
        content = handler(args)
    except Exception as e:                 # 工具自己崩了，别让整个请求挂掉
        log.warning("工具执行失败 tool=%s err=%s", name, e)
        return f"工具执行失败：{e}"

    # 只缓存"成功结果"。失败话术（如"天气服务暂时连不上"）一旦被缓存，
    # 接下来 5 分钟每次都会返回同一条失败文案——实测踩过这个坑。
    if not content.startswith("天气服务暂时连不上"):
        _TOOL_CACHE[key] = (now, content)
    return content


def run_agent(model_messages, max_rounds=MAX_TOOL_ROUNDS, trace=None):
    """跑一次 agent 循环。返回 (answer, used_tool, rounds)。

    answer 为 None 表示"到轮数上限还没收口"。
    trace：可选的 list，传进来就往里追加事件（D3 用），不传就什么也不记。
    """
    used_tool = False  # 初始时先假设没用过工具
    for rnd in range(1, max_rounds + 1):
        _t_llm = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=model_messages,
            temperature=MODEL_TEMPERATURE,
            tools=tools,            # 每一轮都带 tools，模型才能连续调
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if trace is not None:
            trace.append({
                "kind": "llm",
                "round": rnd,
                "ms": round((time.perf_counter() - _t_llm) * 1000),
                "tool_calls": len(msg.tool_calls or []),
            })

        if not msg.tool_calls:
            return msg.content, used_tool, rnd     # 收口：模型不再要工具

        used_tool = True
        model_messages.append(msg)                 # 把"我要调工具"这步记进上下文
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            _t_tool = time.perf_counter()
            content = call_tool(name, args, trace)
            if trace is not None:
                trace.append({
                    "kind": "tool",
                    "name": name,
                    "args": args,
                    "ms": round((time.perf_counter() - _t_tool) * 1000),
                    "result_len": len(str(content)),
                })
            model_messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": name,
                "content": content,
            })

    return None, used_tool, max_rounds


# D4 前置门：不是所有"多轮问题"都需要改写。
# 判据只有两类：句子里有指代（它/那个/刚才…），或者短到必须靠上文才懂（"那学费呢"）。
DEICTIC = ("刚才", "上次", "上面", "前面", "这个", "那个", "这些", "那些",
           "它", "该", "此", "还有")
TIME_WORDS = ("今天", "现在", "目前", "最近", "最新", "今年", "马上")


# 同音/形近错别字：不依赖上文，任何时候都该修（"报道"→"报到"这类）
TYPOS = {
    # 同音/形近错别字
    "报道": "报到", "帐号": "账号", "宿社": "宿舍", "图书管": "图书馆",
    # 口语 → 文档用词（库里写"食堂"，学生说"饭堂"）
    "饭堂": "食堂", "寝室": "宿舍", "宿舍楼": "宿舍",
}


def fix_typos(text: str) -> str:
    for wrong, right in TYPOS.items():
        if wrong in text:
            text = text.replace(wrong, right)
    return text


def need_rewrite(query: str, has_history: bool) -> bool:
    """要不要为了检索去改写这句话。False 时一句话都不多花。"""
    if not has_history:
        return False                       # 第一轮，没有可比对的上文
    q = query.strip()
    if any(w in q for w in DEICTIC):
        return True                        # 有指代，必须补全
    if q.startswith(("那", "这", "它", "还有")):
        return True                        # "那学费呢"这类省略句
    if q.endswith("呢") and len(q) <= 8:
        return True                        # "深圳的呢"——靠上文才懂
    return False                           # 自带实体、能独立检索 -> 不改


def rewrite_ok(query: str, rewritten: str) -> bool:
    """改写结果的出口校验：不合格就退回原句。"""
    if not (2 <= len(rewritten) <= 60):
        return False
    if any(w in rewritten for w in DEICTIC):
        return False                       # 还有指代 = 没改完
    for w in TIME_WORDS:
        if w in rewritten and w not in query:
            return False                   # 引入了原句没有的时间词
    return True


def rewrite_query(query: str, session_id: str, trace=None) -> str:
    """把"刚才那个电话"这类带指代的问题补全成可检索的问句。

    没有历史就原样返回（第一轮没必要多花一次模型调用）。
    改写失败也原样返回——改写是优化，不能变成故障点。
    """
    query = fix_typos(query or "")      # 错别字不依赖历史，先修
    if not ENABLE_QUERY_REWRITE:
        return query

    hist = load_history(session_id, limit=4)
    summary_text = load_summary(session_id)
    if not need_rewrite(query, bool(hist or summary_text)):
        return query

    lines = []
    if summary_text:
        lines.append(f"（更早的对话摘要）{summary_text}")
    for m in hist:
        who = "用户" if m["role"] == "user" else "助手"
        lines.append(f"{who}: {m['content'][:120]}")
    history_text = "\n".join(lines[-6:]) or "（无）"

    _t = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user",
                       "content": REWRITE_PROMPT.format(history=history_text, query=query)}],
            temperature=0,
        )
        rewritten = (resp.choices[0].message.content or "").strip().strip('"').strip("“”")
    except Exception as e:
        log.warning("查询改写失败，用原句 session=%s err=%s", session_id, e)
        return query

    # 出口校验：太短太长、还有指代、凭空加了时间词 -> 一律退回原句
    if not rewrite_ok(query, rewritten):
        log.info("改写结果不合格，退回原句 「%s」 -> 「%s」", query, rewritten)
        return query

    if rewritten != query:
        log.info("查询改写 %s -> %s", query, rewritten)
        if trace is not None:
            trace.append({
                "kind": "rewrite",
                "from": query,
                "to": rewritten,
                "ms": round((time.perf_counter() - _t) * 1000),
            })
    return rewritten


def build_messages(session_id: str, query: str, context: str) -> list[dict]:
    system_content = SYSTEM_PROMPT_TMPL.format(context=context)
    summary_text = load_summary(session_id)
    if summary_text:
        # 摘要放 system 里，和检索到的资料一起给模型
        system_content += f"\n\n【此前对话摘要】{summary_text}"
    messages = [{"role": "system", "content": system_content}]
    messages.extend(load_history(session_id))
    messages.append({"role": "user", "content": query})
    return messages


# ----------------------------------------------------------------------------
# 接口
# ----------------------------------------------------------------------------
@app.get("/traces/{session_id}")
async def traces(session_id: str, limit: int = 20):
    """回放某个会话最近的请求轨迹（最新的在前）。"""
    return {"session_id": session_id, "traces": get_traces(session_id, limit)}


@app.get("/health")
async def health():
    """健康检查：部署时用来确认服务活着。"""
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    # 注意：这里不能因为"知识库没资料"就直接兜底。
    # 天气这类问题本来就不在知识库里，答案是靠工具查的，
    # 提前返回会让工具永远没机会被调用。兜底放在"模型没调工具"之后。
    trace: list[dict] = []
    search_query = rewrite_query(query, request.session_id, trace)
    _t_ret = time.perf_counter()
    context = get_school_context(search_query, fallback=query)
    trace.append({
        "kind": "retrieval",
        "hits": context.count("\n---\n") + 1 if context.strip() else 0,
        "chars": len(context),
        "ms": round((time.perf_counter() - _t_ret) * 1000),
    })
    messages = build_messages(request.session_id, query, context)
    max_retries = 3

    for attempt in range(max_retries):
        try:
            answer, used_tool, rounds = run_agent(messages, trace=trace)     #这个message就是model——message
            log.info(
                "agent 完成 rounds=%d used_tool=%s session=%s",
                rounds, used_tool, request.session_id,
            )

            # 模型判断"不用调工具"→代码校验：天气类问题它是不是漏了一次该做的调用？
            # 非流式路径原来没有这一层，实测"今天天气怎么样呀"会被模型凭常识编成
            # "桂林今天多云，气温10℃-20℃"（真实数据是单点值，不存在区间）。
            if not used_tool and wants_weather(query, request.session_id):
                fb = try_answer_weather(query, force=True, session_id=request.session_id)
                if fb is not None and not fb.startswith("你想查哪个城市"):
                    log.info("模型未调工具，代码补调天气（非流式） session=%s", request.session_id)
                    messages[0]["content"] += (
                        "\n\n【已获取的实时数据】\n" + fb
                        + "\n请直接根据以上数据回答用户，不要复述查询过程。"
                    )
                    answer, used_tool, rounds = run_agent(messages, trace=trace)

            if answer is None:
                # 到轮数上限还没收口：当作答不出来，别再让它继续调
                log.warning("agent 达到轮数上限 session=%s", request.session_id)
                answer = "我查了几次还是没查准，你能说得更具体一点吗？"
            elif not used_tool and not context.strip() and not load_summary(request.session_id):
                # 没调工具、知识库没资料、也完全没有历史摘要：这才说明我们答不了。
                # 有摘要时必须放行——用户可能在问上下文里早就说过的事（"刚才那个电话是多少"）。
                log.info("无资料且未调用工具，返回兜底 session=%s", request.session_id)
                trace.append({"kind": "no_data", "reason": "no_context_no_tool_no_summary"})
                save_trace(request.session_id, trace)
                return {"answer": NO_DATA_ANSWER}

            final_answer = answer

            if not final_answer:
                final_answer = "抱歉，我这边没有生成有效回答，请再问一次。"

            # 写入记忆并裁剪
            append_message(request.session_id, "user", query)
            append_message(request.session_id, "assistant", final_answer)
            maybe_compress(request.session_id)

            trace.append({
                "kind": "answer",
                "chars": len(final_answer),
                "rounds": rounds,
                "used_tool": used_tool,
            })
            save_trace(request.session_id, trace)
            return {"answer": final_answer}

        except RateLimitError:
            if attempt == max_retries - 1:
                log.warning("触发限流且重试耗尽 session=%s", request.session_id)
                raise HTTPException(status_code=429, detail="请求太频繁，请稍后再试")
            wait = 2 ** attempt
            log.warning("触发限流，%d 秒后重试（第 %d 次）", wait, attempt + 1)
            time.sleep(wait)

        except HTTPException:
            raise

        except Exception:
            log.exception("处理请求失败 session=%s", request.session_id)
            raise HTTPException(status_code=500, detail="服务内部错误，请稍后再试")


@app.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    # 和 /chat 一样：不能因为知识库没资料就提前返回，否则工具永远调不到。
    # 兜底放在"模型决定不调工具"之后（见 generate 里的 elif）。
    # 纯城市名追问（上一轮在聊天气，这一轮只回"桂林"）：意图唯一，直接查，不劳模型。
    # 实测这种句子交给模型时，它有时调了工具却仍回"没有相关资料"。
    if wants_weather(query, request.session_id) and not _WEATHER_RE.search(query):
        fb0 = try_answer_weather(query, force=True, session_id=request.session_id)
        if fb0 is not None:
            append_message(request.session_id, "user", query)
            append_message(request.session_id, "assistant", fb0)
            def stream_city_only():
                yield f"data: {fb0}\n\n"
            return StreamingResponse(stream_city_only(), media_type="text/event-stream")

    search_query = rewrite_query(query, request.session_id)
    context = get_school_context(search_query, fallback=query)
    messages = build_messages(request.session_id, query, context)

    def generate():
        full_answer = ""
        try:
            # 第一阶段：非流式，判断要不要调工具
            # （流式响应里拿不到完整的 tool_calls，所以必须先用非流式问一次）
            # 流式路径只挂"外部实时数据"类工具。search_school 的结果本该走 context（前面已检索过），
            # 再当工具结果塞进 system 会和 context 打架，模型就会答"没有相关资料"。
            stream_tools = [
                t for t in tools
                if t["function"]["name"] in ("get_weather", "get_current_time")
            ]
            first = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, temperature=MODEL_TEMPERATURE,
                tools=stream_tools, tool_choice="auto",
            )
            response_message = first.choices[0].message

            if response_message.tool_calls:
                # 第二阶段不传 tools：历史里若留着结构化的 tool_calls，
                # 模型看到"有调用但没有工具定义"，会把调用原样复述成文本（页面上会漏出 JSON）。
                # 所以这里把"调用 + 结果"压成一条普通人话再发。
                # 把工具结果写进 system，而不是伪造一条 assistant 消息。
                # 原因：messages 若以 assistant 结尾，模型会认为"轮到用户说了"，直接返回空。
                tool_notes = []
                called_tools: set[str] = set()
                for tool_call in response_message.tool_calls:
                    name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    # 只放"结果本身"，不要放 get_weather({...}) 这种调用格式——
                    # 模型会把它当成"我要复述的调用"照抄到答案里。
                    tool_notes.append(str(call_tool(name, args)))
                    called_tools.add(name)

                # 模型调了工具，但可能调的不是该调的那个（实测：问天气它去调 get_current_time）。
                # 意图是天气、却没调 get_weather → 补一次，别让它拿别的工具结果编天气。
                if wants_weather(query, request.session_id) and "get_weather" not in called_tools:
                    fb2 = try_answer_weather(query, force=True, session_id=request.session_id)
                    if fb2 is not None:
                        log.info("模型调错工具，代码补调天气 session=%s", request.session_id)
                        tool_notes.append(str(fb2))
                messages[0]["content"] += (
                    "\n\n【已获取的实时数据】\n" + "\n".join(tool_notes)
                    + "\n请直接根据以上数据回答用户，不要复述查询过程，不要输出函数名或参数。"
                )
                yield "data: [系统提示：正在查询实时信息...]\n\n"
            else:
                # 模型判断"不用调工具" → 代码校验：它是不是漏了一次该做的工具调用？
                # 只对"有确定性数据源"的天气类做，且只补调一次。
                fb = try_answer_weather(query, force=True, session_id=request.session_id) if wants_weather(query, request.session_id) else None
                if fb is not None and fb.startswith("你想查哪个城市"):
                    # 生成器里不能 return 一个 StreamingResponse（那只是结束生成器，客户端收不到内容）
                    append_message(request.session_id, "user", query)
                    append_message(request.session_id, "assistant", fb)
                    yield f"data: {fb}\n\n"
                    return
                if fb is not None:
                    log.info("模型未调工具，代码补调天气 session=%s", request.session_id)
                    messages[0]["content"] += (
                        "\n\n【已获取的实时数据】\n" + fb
                        + "\n请直接根据以上数据回答用户，不要复述查询过程。"
                    )

            if not response_message.tool_calls and not context.strip() and not load_summary(request.session_id) and not wants_weather(query, request.session_id):
                log.info("无资料且未调用工具，流式返回兜底 session=%s", request.session_id)
                yield f"data: {NO_DATA_ANSWER}\n\n"
                return

            # 第二阶段：带上工具结果，开启流式
            stream = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, temperature=MODEL_TEMPERATURE, stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_answer += delta
                    yield f"data: {delta}\n\n"

            append_message(request.session_id, "user", query)
            append_message(request.session_id, "assistant", full_answer)
            maybe_compress(request.session_id)
            log.info("流式回答完成 session=%s 长度=%d", request.session_id, len(full_answer))

        except Exception:
            log.exception("流式回答失败 session=%s", request.session_id)
            yield "data: [ERROR] 服务出错了，请稍后再试\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# 前端页面（必须在所有接口之后挂载）
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

