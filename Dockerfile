FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements.txt

# Install dependencies (added --no-cache-dir to keep the image size small)
RUN pip install pip --upgrade && pip install --no-cache-dir -r requirements.txt

COPY . .


ENV GRADIO_SERVER_NAME="0.0.0.0"

EXPOSE 7860


CMD ["gradio", "main.py"]