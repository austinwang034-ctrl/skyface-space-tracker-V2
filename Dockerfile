# Works on Render (free plan included) and any other Docker host.
FROM python:3.12-slim

# Run as an ordinary user, not root; everything the app writes lives in that user's home.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    SPHEREX_VIEWER_CACHE=/home/user/cache
WORKDIR /home/user/app

COPY --chown=user requirements-server.txt .
RUN pip install --no-cache-dir --user -r requirements-server.txt
COPY --chown=user spherex_viewer ./spherex_viewer
COPY --chown=user server.py LICENSE NOTICE ./
COPY --chown=user web ./web

# Hosts such as Render set $PORT and the server listens there; 7860 is only the fallback.
EXPOSE 7860
# One worker on purpose: jobs are kept in memory.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --proxy-headers --forwarded-allow-ips=*"]
