FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY setup.py .
RUN pip install -e . --quiet

COPY apps/fashion-mnist/requirements.txt .
RUN pip install -r requirements.txt --quiet

COPY apps/fashion-mnist/ ./apps/fashion-mnist/

ENV PYTHONPATH=/app/apps/fashion-mnist
ENTRYPOINT ["python", "apps/fashion-mnist/src/main.py"]
CMD ["etl"]
