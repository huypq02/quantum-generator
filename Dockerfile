FROM python:3.13
WORKDIR /usr/local/app/quantumgenerator

# Copy source code and config first (required by pyproject.toml)
COPY src ./src
COPY config ./config
COPY README.md ./

# Install package and dependencies in editable mode
COPY pyproject.toml ./
RUN pip install -e .

EXPOSE 8080

# Setup an app user so the container doesn't run as the root user
RUN useradd -m app && chown -R app:app /usr/local/app/quantumgenerator

USER app

CMD [ "uvicorn", "quantumgenerator.interfaces.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
