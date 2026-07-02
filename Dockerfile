FROM python:3.12-slim

WORKDIR /app

# Install dependencies AND download spaCy models in a single layer.
# (setup.sh installs deps + all 7 models; the explicit pip/download lines make
#  the core requirements reproducible and are safe to keep.)
COPY requirements.txt setup.sh ./
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download pt_core_news_sm en_core_web_sm \
    && bash setup.sh

# Copy the rest of the project (secrets/venv/db excluded via .dockerignore).
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
