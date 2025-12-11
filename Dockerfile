# Use Python 3.9 slim image
FROM python:3.9-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /home/user/app

# Copy the entire application
COPY --chown=user:user . .

# Install the luma_ge package in development mode (this will install all dependencies from pyproject.toml)
RUN pip install --no-cache-dir -e .

# Get Google Service Account JSON secret and save it to auth directory at buildtime
RUN --mount=type=secret,id=GOOGLE_SERVICE_ACCOUNT_JSON_B64,mode=0444,required=false \
    mkdir -p /home/user/app/auth && \
    if [ -f /run/secrets/GOOGLE_SERVICE_ACCOUNT_JSON_B64 ]; then \
    base64 -d /run/secrets/GOOGLE_SERVICE_ACCOUNT_JSON_B64 > /home/user/app/auth/service-account.json; \
    fi && \
    chown -R user:user /home/user/app/auth

# Create temp directory for leafmap and ensure proper permissions
RUN mkdir -p /home/user/tmp && \
    chown -R user:user /home/user/app && \
    chown -R user:user /home/user/tmp

# Switch to non-root user
USER user

# Set environment variables
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH
ENV TMPDIR=/home/user/tmp
ENV TEMP=/home/user/tmp

# Expose the port for Jupyter Lab
EXPOSE 8888

# Command to run Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]