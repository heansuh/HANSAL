FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

COPY apps/cifar-10/requirements.txt ./apps/cifar-10/requirements.txt
RUN pip install -r apps/cifar-10/requirements.txt --quiet

COPY apps/cifar-10/ ./apps/cifar-10/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "/app/apps/cifar-10/src/main.py"]
