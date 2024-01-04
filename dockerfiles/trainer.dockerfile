# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt requirements.txt
# COPY pyproject.toml pyproject.toml
# COPY mlops_classifier/ mlops_classifier/
# COPY data/ data/
COPY . .

WORKDIR /
RUN pip install . --no-cache-dir

ENTRYPOINT ["python", "-u", "mlops_classifier/train_model.py", "train"]