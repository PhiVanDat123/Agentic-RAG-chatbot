FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000 8501

# Script to run both services
COPY start_services.sh .
RUN chmod +x start_services.sh

CMD ["./start_services.sh"]