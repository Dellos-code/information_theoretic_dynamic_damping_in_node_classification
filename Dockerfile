FROM python:3.11-slim

WORKDIR /workspace
COPY pyproject.toml .
COPY edsp edsp
COPY tests tests
RUN pip install --no-cache-dir ".[test]"
CMD ["python", "-m", "pytest", "-q", "tests/unit"]
