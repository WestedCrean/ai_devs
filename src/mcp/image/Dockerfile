FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY . .
RUN uv sync --no-dev
EXPOSE 8000
CMD ["uv", "run", "python", "main.py"]
