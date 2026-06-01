FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt 

EXPOSE 8000
EXPOSE 8501

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0"]