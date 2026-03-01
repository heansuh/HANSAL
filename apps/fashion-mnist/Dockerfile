FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt --quiet

COPY . /app/apps/fashion-mnist/

ENV PYTHONPATH=/app

CMD ["python", "/app/apps/fashion-mnist/src/main.py"]
