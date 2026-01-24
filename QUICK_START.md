# Quick Start für Professoren

## 🎯 Einfachste Methode: Cloud Deployment (Keine Installation!)

### Railway.app (Empfohlen)

1. **Gehe zu:** https://railway.app
2. **Klicke:** "New Project" → "Deploy from GitHub repo"
3. **Wähle:** Dieses Repository (`schmiroBOS99/realtime-automated-factchecker`)
4. **Warte:** 5-10 Minuten (Railway baut automatisch das Docker-Image)
5. **Fertig!** Service ist live unter einer Railway-URL (z.B. `https://factchecker-production.up.railway.app`)

**Vorteile:**
- ✅ Keine Installation nötig
- ✅ Direkt im Browser testbar
- ✅ Automatische Updates bei Git Push
- ✅ Kostenlos für kleine Projekte

### Alternative: Render.com

1. **Gehe zu:** https://render.com
2. **Klicke:** "New" → "Web Service"
3. **Verbinde:** GitHub Repository
4. **Wähle:** Branch `main`
5. **Build Command:** (automatisch erkannt)
6. **Start Command:** (automatisch erkannt)
7. **Fertig!** Service ist live unter einer Render-URL

## 🐳 Alternative: Docker Compose (Lokal)

**Voraussetzung:** Docker Desktop installiert (https://www.docker.com/products/docker-desktop)

```bash
# Repository klonen
git clone https://github.com/schmiroBOS99/realtime-automated-factchecker.git
cd realtime-automated-factchecker

# Service starten
docker-compose up
```

**Service erreichbar unter:** http://localhost:8000

## 📱 UI verwenden

1. Öffne die URL im Browser (Railway/Render URL oder http://localhost:8000)
2. Die Fact-Checker UI wird automatisch geladen
3. Gib einen Claim ein (z.B. "Berlin ist die Hauptstadt von Deutschland")
4. Klicke auf "Check" oder nutze die Live-Transkription

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
- Siehe [README_DEPLOYMENT.md](README_DEPLOYMENT.md) für Deployment-Optionen
- Öffne ein Issue auf GitHub
