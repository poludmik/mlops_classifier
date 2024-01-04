# Base image
# FROM python:3.9-slim
FROM  nvcr.io/nvidia/pytorch:22.07-py3

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt requirements.txt
# COPY pyproject.toml pyproject.toml
# COPY mlops_classifier/ mlops_classifier/
# COPY data/ data/

WORKDIR /usr/local/
COPY . .

RUN pip install . --no-cache-dir

ENTRYPOINT ["python", "-u", "mlops_classifier/predict_model.py", "predict", "models/MyAwesomeModel/model.pt", "data/raw/random_mnist_images/mnist_sample.npy"]
