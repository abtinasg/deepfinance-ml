# DeepFinance ML Engine - Optimized Dockerfile
# Multi-stage build for Railway deployment (<1GB final image)

# =============================================================================
# Stage 1: Builder - Install dependencies and build wheels
# =============================================================================
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch CPU-only first (saves ~1.5GB)
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install XGBoost minimal (CPU-only, no CUDA)
RUN pip install --no-cache-dir xgboost==2.0.3

# Install remaining production dependencies (excluding dev tools)
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    pydantic==2.5.3 \
    pydantic-settings==2.1.0 \
    httpx==0.26.0 \
    numpy==1.26.3 \
    pandas==2.1.4 \
    scikit-learn==1.4.0 \
    yfinance==0.2.35 \
    redis==5.0.1 \
    slowapi==0.1.9 \
    prometheus-client==0.19.0 \
    python-dotenv==1.0.0

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM python:3.11-slim AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install only runtime system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN adduser --disabled-password --gecos '' --uid 1000 appuser

# Copy only necessary application files
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser services/ ./services/
COPY --chown=appuser:appuser config.py ./config.py
COPY --chown=appuser:appuser main.py ./main.py
COPY --chown=appuser:appuser README.md ./README.md

# Switch to non-root user
USER appuser

# Expose port (Railway uses PORT env variable)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/v1/health || exit 1

# Run the application
CMD ["python", "main.py"]
