# 🤖 RAG Assistant - Guide d'Installation

Un système RAG (Retrieval-Augmented Generation) complet avec interface web moderne utilisant les couleurs de votre logo.

## ✨ Fonctionnalités

- **Upload de documents** : PDF, DOCX, TXT
- **Interface responsive** avec couleurs du logo (bleu, teal, vert)
- **API compatible OpenAI** pour l'intégration
- **Clavier virtuel** pour tablettes
- **Recherche vectorielle** avec ChromaDB
- **Interface moderne** et intuitive

## 🚀 Installation

### 1. Prérequis
```bash
# Python 3.8+
python --version

# Créer un environnement virtuel
python -m venv rag_env
source rag_env/bin/activate  # Linux/Mac
# ou
rag_env\Scripts\activate     # Windows
```

### 2. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 3. Structure des fichiers
```
rag_project/
├── main.py              # Serveur backend
├── requirements.txt     # Dépendances Python
├── index.html          # Interface frontend
├── uploads/            # Dossier pour les fichiers (créé auto)
├── chroma_db/          # Base vectorielle (créé auto)
└── README.md           # Ce fichier
```

## 🔧 Configuration

### Variables d'environnement (optionnel)
```bash
# .env
OPENAI_API_KEY=your_openai_key_here
MISTRAL_API_KEY=your_mistral_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
```

### Personnalisation de l'API
Dans `main.py`, modifiez la fonction `generate_response()` pour utiliser votre API préférée :

```python
async def generate_response(system_prompt: str, user_message: str, context: str) -> str:
    # Exemple avec OpenAI
    import openai
    openai.api_key = "your-api-key"
    
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content
```

## 🚀 Démarrage

### 1. Lancer le serveur backend
```bash
python main.py
```
Le serveur démarre sur `http://localhost:8000`

### 2. Accéder à l'interface
Ouvrez votre navigateur sur `http://localhost:8000` ou servez le fichier `index.html`

## 📱 Utilisation

### Upload de documents
1. Cliquez sur la zone d'upload ou glissez-déposez vos fichiers
2. Formats supportés : PDF, DOCX, TXT
3. Les documents sont automatiquement traités et indexés

### Chat avec l'IA
1. Tapez votre question dans la zone de texte
2. L'IA recherche dans vos documents uploadés
3. Recevez une réponse basée sur le contenu de vos documents

### Tablettes
- Le clavier virtuel s'affiche automatiquement sur tablette
- Interface responsive pour tous les écrans
- Optimisé pour le tactile

## 🔌 API Endpoints

### Upload de fichiers
```bash
POST /upload
Content-Type: multipart/form-data
Body: file (PDF/DOCX/TXT)
```

### Chat (compatible OpenAI)
```bash
POST /v1/chat/completions
Content-Type: application/json
Body: {
  "messages": [{"role": "user", "content": "votre question"}],
  "model": "gpt-3.5-turbo"
}
```

### Liste des documents
```bash
GET /documents
```

### Supprimer tous les documents
```bash
DELETE /documents
```

### Santé de l'API
```bash
GET /health
```

## 🎨 Personnalisation des couleurs

Les couleurs sont définies dans les variables CSS :
```css
:root {
    --primary-blue: #1e3a8a;     /* Bleu principal */
    --secondary-teal: #0891b2;    /* Teal secondaire */
    --accent-green: #10b981;      /* Vert accent */
    --gradient-bg: linear-gradient(135deg, #1e3a8a 0%, #0891b2 50%, #10b981 100%);
}
```

## 🔧 Intégration avec d'autres APIs

### OpenAI
```python
import openai
openai.api_key = "votre-clé"
```

### Mistral
```python
from mistralai.client import MistralClient
client = MistralClient(api_key="votre-clé")
```

### Deepseek
```python
import requests
headers = {"Authorization": "Bearer votre-clé"}
```

### OpenRouter
```python
import requests
headers = {"Authorization": "Bearer votre-clé"}
url = "https://openrouter.ai/api/v1/chat/completions"
```

## 🐛 Dépannage

### Erreur de port
```bash
# Changer le port dans main.py
uvicorn.run("main:app", host="0.0.0.0", port=8001)
```

### Problème CORS
Vérifiez que l'URL de l'API dans `index.html` correspond :
```javascript
const API_BASE = 'http://localhost:8000';
```

### Erreur d'embedding
```bash
# Vider le cache des modèles
rm -rf ~/.cache/huggingface/
```

## 📝 Logs
Les logs sont affichés dans la console du serveur pour le débogage.

## 🛡️ Sécurité
- Ajoutez l'authentification pour la production
- Limitez les types de fichiers acceptés
- Validez la taille des fichiers
- Utilisez HTTPS en production

## 🚀 Déploiement en production

### Docker
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
```

### Variables d'environnement de production
```bash
export HOST=0.0.0.0
export PORT=8000
export WORKERS=4
```

Votre système RAG est maintenant prêt ! 🎉