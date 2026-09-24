# Utilise une image Python officielle et légère
FROM python:3.11-slim-bookworm

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Installe les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copie le fichier des dépendances et les installe
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le reste du code
COPY . .

# Expose le port sur lequel l'application va tourner
EXPOSE 8000

# La commande pour lancer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
