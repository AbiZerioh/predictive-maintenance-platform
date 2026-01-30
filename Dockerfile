FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Créer les répertoires nécessaires
RUN mkdir -p data/raw data/processed models logs

# Exposer les ports
EXPOSE 8000 8501

# Commande par défaut (sera override par docker-compose)
CMD ["python", "-c", "print('Predictive Maintenance Platform ready')"]
