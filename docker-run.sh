#!/bin/bash

# TikZ Agent Docker Runner Script
# This script makes it easy to run the TikZ Agent Docker container

set -e

# Default values
PORT=${PORT:-8501}
IMAGE_NAME=${IMAGE_NAME:-tikz-agent}

echo "🐳 TikZ Agent Docker Runner"
echo "=========================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if image exists, if not build it
if ! docker image inspect $IMAGE_NAME &> /dev/null; then
    echo "🔨 Building Docker image..."
    docker build -t $IMAGE_NAME .
fi

# Check for API keys
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ]; then
    echo "⚠️  Warning: No API keys found in environment variables."
    echo "   Please set at least one of:"
    echo "   - OPENAI_API_KEY"
    echo "   - ANTHROPIC_API_KEY" 
    echo "   - GOOGLE_API_KEY"
    echo ""
    echo "   Example:"
    echo "   export OPENAI_API_KEY='your_key_here'"
    echo "   ./docker-run.sh"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create output directory if it doesn't exist
mkdir -p output

echo "🚀 Starting TikZ Agent on port $PORT..."
echo "   Open http://localhost:$PORT in your browser"
echo "   Press Ctrl+C to stop"
echo ""

# Run the container
docker run --rm -it \
    -p $PORT:$PORT \
    -e PORT=$PORT \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
    -e GOOGLE_API_KEY="$GOOGLE_API_KEY" \
    -v $(pwd)/output:/app/output \
    --name tikz-agent-temp \
    $IMAGE_NAME 