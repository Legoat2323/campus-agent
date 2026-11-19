# 校园问答 Agent —— 生产镜像
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=utf-8 \
    TZ=Asia/Shanghai

# pip 源：默认官方；国内慢时用 --build-arg PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_INDEX=https://pypi.org/simple

# 先装依赖：这一层能被缓存，改代码时不用重装
COPY requirements.txt .
RUN pip install -i ${PIP_INDEX} -r requirements.txt

# 再拷代码和数据
COPY main.py weather.py ./
COPY static/ ./static/
COPY data/ ./data/
COPY chroma_db/ ./chroma_db/

EXPOSE 8000

# 容器里必须监听 0.0.0.0，不能是 127.0.0.1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
