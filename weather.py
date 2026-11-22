
"""两个坑（实测踩到的）：
    1. Open-Meteo 的地理编码对中文地名很差。
       直接搜「桂林」返回的是安徽/福建/台湾的同名小村庄，人口全是 0。
       搜「Guilin」才能拿到桂林市（广西，人口 157 万）。
       → 所以内置一张常用城市坐标表，命中就用，命中不了再回落。
    2. weather_code 是 WMO 数字码，要翻译成人话。
"""
import json
import time
import urllib.parse
import urllib.request

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# 常用城市 → 英文名（用于地理编码）
CITY_ALIAS = {
    "桂林": "Guilin", "南宁": "Nanning", "柳州": "Liuzhou",
    "北京": "Beijing", "上海": "Shanghai", "广州": "Guangzhou",
    "深圳": "Shenzhen", "杭州": "Hangzhou", "成都": "Chengdu",
    "重庆": "Chongqing", "武汉": "Wuhan", "西安": "Xi'an",
    "南京": "Nanjing", "天津": "Tianjin", "长沙": "Changsha",
    "昆明": "Kunming", "贵阳": "Guiyang", "海口": "Haikou",
    "郑州": "Zhengzhou", "南昌": "Nanchang",
}

# 常用城市坐标（纬度, 经度, 规范名）：命中就跳过 geocoding 请求，省一次网络往返
CITY_COORD = {
    "桂林": (25.27, 110.29, "桂林"), "南宁": (22.82, 108.32, "南宁"),
    "柳州": (24.31, 109.41, "柳州"), "北京": (39.90, 116.41, "北京"),
    "上海": (31.23, 121.47, "上海"), "广州": (23.13, 113.26, "广州"),
    "深圳": (22.54, 114.06, "深圳"), "杭州": (30.27, 120.16, "杭州"),
    "成都": (30.57, 104.07, "成都"), "重庆": (29.56, 106.55, "重庆"),
    "武汉": (30.59, 114.31, "武汉"), "西安": (34.34, 108.94, "西安"),
    "南京": (32.06, 118.80, "南京"), "天津": (39.13, 117.20, "天津"),
    "长沙": (28.23, 112.94, "长沙"), "昆明": (25.04, 102.71, "昆明"),
    "贵阳": (26.65, 106.63, "贵阳"), "海口": (20.04, 110.32, "海口"),
    "郑州": (34.75, 113.62, "郑州"), "南昌": (28.68, 115.86, "南昌"),
}

# WMO 天气码 → 中文
WEATHER_CODE = {
    0: "晴", 1: "晴间多云", 2: "多云", 3: "阴",
    45: "有雾", 48: "雾凇",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    56: "冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "强阵雪",
    95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "强雷阵雨伴冰雹",
}

# 简单缓存：同一个城市 10 分钟内不重复请求
_cache: dict[str, tuple[float, str]] = {}
CACHE_TTL = 600


def _get_json(url: str, params: dict, timeout: int = 6) -> dict:
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"{url}?{query}", headers={"User-Agent": "campus-agent/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _geocode(city: str) -> tuple[float, float, str] | None:
    """返回 (纬度, 经度, 规范城市名)。内置坐标优先，查不到再走网络。"""
    if city in CITY_COORD:
        return CITY_COORD[city]
    name = CITY_ALIAS.get(city, city)
    try:
        data = _get_json(GEO_URL, {"name": name, "count": 5, "language": "zh"}, timeout=5)
    except Exception:
        return None

    results = data.get("results") or []
    # 优先选中国的、人口最多的
    chinese = [r for r in results if r.get("country") == "中国"]
    pool = chinese or results
    if not pool:
        return None
    best = max(pool, key=lambda r: r.get("population") or 0)
    return best["latitude"], best["longitude"], best.get("name", city)


def get_weather(city: str) -> str:
    """查某个城市的实时天气，返回一句中文描述。

    失败时返回可读的提示，不抛异常——工具调用里抛异常会让整个对话中断。
    """
    city = (city or "").strip()
    if not city:
        return "没有识别出城市名，请告诉我要查哪个城市。"

    now = time.time()
    if city in _cache and now - _cache[city][0] < CACHE_TTL:
        return _cache[city][1]

    loc = _geocode(city)
    if not loc:
        return f"没有找到「{city}」这个城市，换个说法试试？"

    lat, lon, real_name = loc
    try:
        data = _get_json(
            WEATHER_URL,
            {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "timezone": "Asia/Shanghai",
            },
        )
    except Exception:
        return f"天气服务暂时连不上，稍后再问「{city}」的天气吧。"

    cur = data.get("current") or {}
    if not cur:
        return f"没有拿到「{city}」的天气数据。"

    desc = WEATHER_CODE.get(cur.get("weather_code"), "未知天气")
    temp = cur.get("temperature_2m")
    humid = cur.get("relative_humidity_2m")
    wind = cur.get("wind_speed_10m")

    text = (
        f"{real_name}现在{desc}，气温 {temp}℃，"
        f"湿度 {humid}%，风速 {wind} km/h。"
    )
    _cache[city] = (now, text)
    return text


if __name__ == "__main__":
    for c in ["桂林", "北京", "南宁", "不存在的城市"]:
        print(f"{c:<8} -> {get_weather(c)}")
