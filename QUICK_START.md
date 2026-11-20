# Quick Start

## 🐳 Docker Compose (Empfohlene Methode)

### Voraussetzung: Docker Desktop installieren

1. **Gehen Sie zu:** https://www.docker.com/products/docker-desktop
2. **Laden Sie Docker Desktop herunter** und installieren Sie es
3. **Starten Sie Docker Desktop** (muss laufen, bevor Sie den Service starten)

### Service starten

```bash
# 1. Repository klonen
git clone https://github.com/schmiroBOS99/realtime-automated-factchecker.git
cd realtime-automated-factchecker

# 2. Service starten
docker-compose up
```

**Service erreichbar unter:** http://localhost:8000

> ⚠️ **Beim ersten Start:** Docker lädt alle Modelle herunter (5-10 Minuten). Lassen Sie das Terminal-Fenster geöffnet!

## 📱 UI verwenden

1. Öffnen Sie die URL im Browser (http://localhost:8000)
2. Die Fact-Checker UI wird automatisch geladen
3. Geben Sie einen Claim ein (z.B. "Berlin ist die Hauptstadt von Deutschland")
4. Klicken Sie auf "Check" oder nutzen Sie die Live-Transkription

## 🔧 API direkt testen

```bash
# Health Check
curl http://localhost:8000/ping

# Claim prüfen
curl -X POST http://localhost:8000/check \
     -H "Content-Type: application/json" \
     -d '{"claim": "Berlin ist die Hauptstadt von Deutschland"}'
```

## ❓ Hilfe

Bei Problemen:
- Siehe [README.md](README.md) für detaillierte Dokumentation
- Öffnen Sie ein Issue auf GitHub
