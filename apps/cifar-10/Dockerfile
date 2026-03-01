FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

COPY apps/cifar-10/requirements.txt .
RUN pip install -r requirements.txt --quiet

COPY apps/cifar-10/ ./apps/cifar-10/

ENV PYTHONPATH=/app/apps/cifar-10
ENTRYPOINT ["python", "apps/cifar-10/src/main.py"]