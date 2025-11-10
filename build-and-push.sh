#!/bin/bash

# Build and push script for Linux amd64
# This script builds for linux/amd64 platform and pushes to Docker Hub
# For local Mac development, use: docker-compose build

set -e

# Read version from VERSION file
VERSION=$(cat VERSION | tr -d '[:space:]')

if [ -z "$VERSION" ]; then
    echo "Error: VERSION file is empty or not found"
    exit 1
fi

echo "Building version: $VERSION for linux/amd64"
echo "Repository: dustinng04/rec-model"

# Create buildx builder if it doesn't exist
if ! docker buildx ls | grep -q "multiarch-builder"; then
    echo "Creating buildx builder..."
    docker buildx create --name multiarch-builder --use || true
fi

# Use the builder
docker buildx use multiarch-builder

# Build and push for linux/amd64
docker buildx build \
    --platform linux/amd64 \
    --tag dustinng04/rec-model:$VERSION \
    --tag dustinng04/rec-model:latest \
    --push \
    .

echo "Successfully built and pushed:"
echo "  - dustinng04/rec-model:$VERSION"
echo "  - dustinng04/rec-model:latest"
echo "  Platform: linux/amd64"

