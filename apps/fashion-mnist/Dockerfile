FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

COPY apps/fashion-mnist/requirements.txt .
RUN pip install -r requirements.txt --quiet

COPY apps/fashion-mnist/ ./apps/fashion-mnist/

ENV PYTHONPATH=/app/apps/fashion-mnist
ENTRYPOINT ["python", "apps/fashion-mnist/src/main.py"]
