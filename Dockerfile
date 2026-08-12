FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["/bin/sh", "-c", "unset STREAMLIT_SERVER_PORT && streamlit run pages/Stock_Predictor.py --server.port 8080 --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false"]
