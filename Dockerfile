FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY docsearch ./docsearch
COPY sample_docs ./sample_docs
ENV INDEX_DIR=/data/index INBOX_DIR=/data/inbox
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "docsearch.api:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
