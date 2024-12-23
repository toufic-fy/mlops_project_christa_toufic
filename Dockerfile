# Stage 1: Base image with Python and Poetry
FROM python:3.13-slim AS base

# Set environment variables for Poetry
ENV POETRY_VERSION=1.8.5 \
    POETRY_HOME=/opt/poetry \
    PATH="/opt/poetry/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libpq-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set working directory
WORKDIR /app

# Copy only the Poetry files to leverage Docker caching
COPY pyproject.toml poetry.lock ./

# Install dependencies (without the source code)
RUN poetry install --no-root --only main --no-dev

# Stage 2: Build stage (optional if building assets)
# Uncomment this if you need to build assets or preprocess data
# FROM base AS build
# Copy application source code
# COPY src ./src
# Add any build/preprocessing commands here

# Stage 3: Final runtime image
FROM python:3.13-slim AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/poetry/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the Poetry environment and app from the base image
COPY --from=base /opt/poetry /opt/poetry
COPY --from=base /app /app

# Set working directory
WORKDIR /app

# Copy application source code
COPY src ./src

# Expose the port used by FastAPI
EXPOSE 8000

# Command to run the app
CMD ["poetry", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
