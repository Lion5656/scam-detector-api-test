# 建立docker image(鏡像)

# 選擇base image(需符合 pyproject.toml 的 requires-python)
FROM python:3.14-slim

# 安裝 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# HF Space 以 uid 1000 執行,工作目錄需屬於該使用者才能寫入模型與資料
RUN useradd -m -u 1000 user && mkdir -p /home/user/app && chown user:user /home/user/app
USER user
WORKDIR /home/user/app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/home/user/app/.venv/bin:$PATH"

# 安裝套件依賴(先只複製依賴檔,讓此層可被快取)
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 搬入原始碼
COPY --chown=user . .

# 執行cmd
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
