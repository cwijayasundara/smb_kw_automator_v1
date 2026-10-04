FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app/backend
COPY packages/model-router /app/packages/model-router
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --extra ocr --extra observability --no-install-project
COPY backend .
RUN uv sync --frozen --no-dev --extra ocr --extra observability
ENV PATH="/app/backend/.venv/bin:$PATH"
EXPOSE 8000
CMD ["keel", "api", "--host", "0.0.0.0"]
