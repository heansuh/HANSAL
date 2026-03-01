FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

<<<<<<< HEAD
COPY requirements.txt .
RUN pip install -r requirements.txt --quiet
=======
COPY apps/fashion-mnist/requirements.txt ./apps/fashion-mnist/requirements.txt
RUN pip install -r apps/fashion-mnist/requirements.txt --quiet
>>>>>>> d088540 (dockerfile updated)

COPY . /app/apps/fashion-mnist/

<<<<<<< HEAD
ENV PYTHONPATH=/app

CMD ["python", "/app/apps/fashion-mnist/src/main.py"]
=======
ENTRYPOINT ["python", "/app/apps/fashion-mnist/src/main.py"]
>>>>>>> d088540 (dockerfile updated)
