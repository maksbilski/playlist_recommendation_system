FROM python:3.10-slim
WORKDIR /app

COPY . .

RUN pip install --no-cache-dir .

ENV FLASK_APP=app.py
ENV PYTHONPATH=/app

EXPOSE 5000

ENTRYPOINT ["python", "./service/app.py"]
CMD ["--train-data", "./data_files/train_sessions.jsonl", \
    "--model-path", "./model_files/wmf_model.pth", \
    "--log-dir", "./log_files"]
