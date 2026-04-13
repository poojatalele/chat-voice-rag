# ── Stage 1: Build React frontend ──────────────────────────────────────────
FROM node:20-slim AS frontend
WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
# No VITE_API_BASE_URL → apiBase() returns "" → same-origin API calls
RUN npm run build

# ── Stage 2: Python backend + serve built frontend ──────────────────────────
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server ./server
COPY scripts ./scripts
COPY data ./data
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Copy the built React app so FastAPI can serve it
COPY --from=frontend /app/web/dist ./web/dist

EXPOSE 8000
CMD ["./entrypoint.sh"]
