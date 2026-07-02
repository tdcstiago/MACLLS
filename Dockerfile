FROM python:3.12-slim

WORKDIR /app

# Install dependencies + all 7 spaCy models in a single cached layer.
# setup.sh runs `pip install -r requirements.txt` and downloads every model.
COPY requirements.txt setup.sh ./
RUN bash setup.sh

# Copy the rest of the project (secrets/venv/db excluded via .dockerignore).
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
