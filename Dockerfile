FROM python:3.10

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
