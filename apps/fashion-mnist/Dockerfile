FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

COPY apps/fashion-mnist/requirements.txt ./apps/fashion-mnist/requirements.txt
RUN pip install -r apps/fashion-mnist/requirements.txt --quiet

COPY apps/fashion-mnist/ ./apps/fashion-mnist/

ENTRYPOINT ["python", "/app/apps/fashion-mnist/src/main.py"]
