# API Assistant Fon

Backend FastAPI pour l'assistant vocal fon. Il expose trois endpoints :
- `/transcribe` : transcription audio fon → texte
- `/translate` : traduction fon → français
- `/speak` : synthèse vocale texte fon → audio

## Architecture

Ce backend est **léger** : il ne charge aucun modèle localement.
Il appelle les API d'inférence de Hugging Face pour chaque tâche.

## Variables d'environnement

- `HF_TOKEN` : ton token Hugging Face (obligatoire)

## Endpoints

| Méthode | Endpoint | Description |
| :--- | :--- | :--- |
| GET | `/` | Statut de l'API |
| POST | `/transcribe` | Upload audio → transcription |
| POST | `/translate` | Texte fon → traduction française |
| POST | `/speak` | Texte fon → audio WAV |

## Déploiement

Déployé automatiquement sur Render à chaque `git push` sur `main`.
