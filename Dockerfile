# FastAPI service.  docker build -t bi-agent . && docker run -p 8000:8000 --env-file .env bi-agent
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf
WORKDIR /app
RUN groupadd -r agent && useradd -r -g agent -d /app agent

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the reranker in so the container starts without downloading it.
ARG BI_RERANKER=cross-encoder/ms-marco-MiniLM-L-6-v2
ENV BI_RERANKER=${BI_RERANKER}
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('${BI_RERANKER}')"
ENV HF_HUB_OFFLINE=1

COPY --chown=agent:agent . .
RUN chown -R agent:agent /app
USER agent
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
