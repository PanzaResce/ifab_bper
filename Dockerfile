FROM python:3.11
WORKDIR /ollama-app

# install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ollama-app/ app/

# execute script
CMD ["python", "app/main.py"]
