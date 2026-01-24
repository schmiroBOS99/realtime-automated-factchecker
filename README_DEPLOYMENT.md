# Deployment Guide

Einfache Anleitung zur Bereitstellung des Fact-Checking Service für Professoren.

## Option 1: Docker Compose (Empfohlen - Einfachste Methode)

### Voraussetzungen
- Docker Desktop installiert: https://www.docker.com/products/docker-desktop

### Starten
```bash
# Repository klonen
git clone https://github.com/schmiroBOS99/realtime-automated-factchecker.git
cd realtime-automated-factchecker

# Service starten (lädt automatisch alle Dependencies)
docker-compose up
```

Der Service ist dann unter **http://localhost:8000** erreichbar.

### Stoppen
```bash
docker-compose down
```

## Option 2: Cloud Deployment (Keine Installation nötig)

### Railway.app (Empfohlen - Kostenlos für kleine Projekte)

1. Gehe zu https://railway.app
2. "New Project" → "Deploy from GitHub repo"
3. Wähle dieses Repository
4. Railway erkennt automatisch das Dockerfile
5. Service ist nach 5-10 Minuten live unter einer Railway-URL

**Vorteil:** Professoren können direkt im Browser testen, keine Installation nötig!

### Alternative: Render.com

1. Gehe zu https://render.com
2. "New" → "Web Service"
3. Verbinde GitHub Repository
4. Build Command: `docker build -t factchecker .`
5. Start Command: `docker run -p 8000:8000 factchecker`

## Option 3: GitHub Codespaces (Im Browser)

1. Gehe zum Repository auf GitHub
2. Klicke auf "Code" → "Codespaces" → "Create codespace"
3. Im Terminal: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
4. Codespace erstellt automatisch eine öffentliche URL

**Vorteil:** Läuft komplett im Browser, keine lokale Installation!

## Option 4: Lokale Installation (Für Entwickler)

Siehe Haupt-README.md für detaillierte Installationsanleitung.

## Empfehlung für Professoren

**Beste Option:** Railway.app oder Render.com Deployment
- Keine Installation nötig
- Direkt im Browser testbar
- Automatische Updates bei Git Push
- Kostenlos für kleine Projekte
