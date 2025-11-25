#!/bin/bash
set -e

# Function to find .env in current or parent directory
find_env_file() {
    if [ -f ".env" ]; then
        echo ".env"
    elif [ -f "../.env" ]; then
        echo "../.env"
    else
        return 1
    fi
}

# Locate .env
ENV_FILE=$(find_env_file) || {
    echo ".env file not found in current or parent directory!"
    exit 1
}

# Load environment variables from .env
export $(grep -v '^#' "$ENV_FILE" | xargs)

# Check if REDIS_PASSWORD is set
if [ -z "$REDIS_PASSWORD" ]; then
    echo "REDIS_PASSWORD is not set in $ENV_FILE"
    exit 1
fi

# Volume name for persistent Redis data
VOLUME_NAME="redis_data"

# Create Docker volume if it doesn't exist
if ! docker volume ls | grep -q "$VOLUME_NAME"; then
    echo "Creating Docker volume: $VOLUME_NAME"
    docker volume create $VOLUME_NAME
fi

# Run Redis container
docker run -d \
  --name redis-server \
  -p 6379:6379 \
  -v $VOLUME_NAME:/data \
  redis:8 \
  redis-server --bind 0.0.0.0 --requirepass $REDIS_PASSWORD