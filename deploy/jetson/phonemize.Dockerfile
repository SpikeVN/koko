FROM python:3.12-slim-bookworm

WORKDIR /app

RUN pip install --no-cache-dir "sea-g2p==0.9.1"

COPY tools/phonemize_server.py ./tools/phonemize_server.py

EXPOSE 8788
CMD ["python", "tools/phonemize_server.py", "--host", "0.0.0.0", "--port", "8788"]
