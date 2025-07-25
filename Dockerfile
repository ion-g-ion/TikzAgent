FROM texlive/texlive:latest

# Set environment variables with defaults
ENV PORT=8501
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Optional API keys (can be overridden at runtime)
ENV OPENAI_API_KEY=""
ENV ANTHROPIC_API_KEY=""
ENV GOOGLE_API_KEY=""

# Install Python 3.11 and essential packages (available in Debian 13)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    curl \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY requirements.txt pyproject.toml ./
COPY TikzAgent/ ./TikzAgent/
COPY streamlit_demo.py run_demo.py ./
COPY README.md LICENSE ./

# Install the project and dependencies
RUN pip install --no-cache-dir -e .[demo]

# Create a non-root user for security (using different UID)
RUN useradd -m -u 1001 tikzuser && chown -R tikzuser:tikzuser /app
USER tikzuser

# Expose the port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/_stcore/health || exit 1

# Run the Streamlit demo
CMD streamlit run streamlit_demo.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true 