FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so code changes do not re-download them
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e ".[dev,dl,app]"

COPY configs ./configs
COPY tests ./tests
COPY app ./app

# Downloaded race data lives in /app/data; mount it so it survives the container:
#   docker run --rm -v "$PWD/data:/app/data" f1pit python -m f1pit.ingest
CMD ["pytest", "-q"]
