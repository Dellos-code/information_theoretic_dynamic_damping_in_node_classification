FROM python:3.11-slim

WORKDIR /workspace
COPY pyproject.toml .
RUN pip install --no-cache-dir numpy scipy pytest hypothesis
COPY edsp edsp
COPY tests tests
CMD ["python", "-m", "pytest", "-q", "tests/unit"]
