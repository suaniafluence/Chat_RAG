import os
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
from docx import Document
import io
import hashlib
import openai
from datetime import datetime
import json
import logging
from huggingface_hub import hf_hub_download

# Configuration
UPLOAD_DIR = "uploads"
CHROMA_DIR = "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"

# Créer les dossiers nécessaires
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation des modèles
@app.on_event("startup")
async def startup_event():
    global embedding_model, chroma_client, collection
    logger.info("Initialisation des modèles...")
    
    # Modèle d'embedding
    embedding_model = SentenceTransformer(MODEL_NAME)
    
    # ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )
    
    logger.info("Modèles initialisés avec succès!")

# Modèles Pydantic
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = False

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict

# Fonctions utilitaires
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extraire le texte d'un fichier PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Erreur extraction PDF: {e}")
        return ""

def extract_text_from_docx(file_content: bytes) -> str:
    """Extraire le texte d'un fichier DOCX"""
    try:
        doc = Document(io.BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Erreur extraction DOCX: {e}")
        return ""

def extract_text_from_txt(file_content: bytes) -> str:
    """Extraire le texte d'un fichier TXT"""
    try:
        return file_content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return file_content.decode('latin-1')
        except Exception as e:
            logger.error(f"Erreur extraction TXT: {e}")
            return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Diviser le texte en chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks

def generate_id(content: str) -> str:
    """Générer un ID unique pour un contenu"""
    return hashlib.md5(content.encode()).hexdigest()

async def search_documents(query: str, k: int = 5) -> List[str]:
    """Rechercher des documents pertinents"""
    try:
        query_embedding = embedding_model.encode([query]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
        
        return results['documents'][0] if results['documents'] else []
    except Exception as e:
        logger.error(f"Erreur recherche: {e}")
        return []

# Endpoints
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload et traitement des fichiers"""
    try:
        # Vérifier le type de fichier
        allowed_extensions = ['.pdf', '.docx', '.txt']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Type de fichier non supporté. Types acceptés: {', '.join(allowed_extensions)}"
            )
        
        # Lire le contenu du fichier
        file_content = await file.read()
        
        # Extraire le texte selon le type de fichier
        if file_extension == '.pdf':
            text = extract_text_from_pdf(file_content)
        elif file_extension == '.docx':
            text = extract_text_from_docx(file_content)
        elif file_extension == '.txt':
            text = extract_text_from_txt(file_content)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Impossible d'extraire le texte du fichier")
        
        # Diviser en chunks
        chunks = chunk_text(text)
        
        # Générer les embeddings et stocker
        embeddings = embedding_model.encode(chunks).tolist()
        
        # Préparer les métadonnées
        ids = [f"{file.filename}_{i}_{generate_id(chunk)}" for i, chunk in enumerate(chunks)]
        metadatas = [{"filename": file.filename, "chunk_id": i} for i in range(len(chunks))]
        
        # Ajouter à ChromaDB
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        return {
            "message": f"Fichier {file.filename} traité avec succès",
            "chunks_count": len(chunks),
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"Erreur upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Endpoint compatible OpenAI pour le chat avec RAG"""
    try:
        # Extraire la dernière question de l'utilisateur
        user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Aucun message utilisateur trouvé")
        
        # Rechercher des documents pertinents
        relevant_docs = await search_documents(user_message)
        
        # Construire le contexte
        context = "\n\n".join(relevant_docs) if relevant_docs else ""
        
        # Construire le prompt avec contexte
        system_prompt = f"""Tu es un assistant IA qui répond aux questions en utilisant les informations fournies dans le contexte.
        
Contexte:
{context}

Instructions:
- Réponds uniquement en te basant sur les informations du contexte fourni
- Si l'information n'est pas dans le contexte, dis-le clairement
- Sois précis et concis
- Cite les sources quand c'est pertinent"""

        # Simuler une réponse (remplacez par votre API préférée)
        response_content = await generate_response(system_prompt, user_message, context)
        
        # Format de réponse compatible OpenAI
        response = ChatResponse(
            id=f"chatcmpl-{generate_id(user_message)}",
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(response_content.split()),
                "total_tokens": len(user_message.split()) + len(response_content.split())
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Erreur chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_response(system_prompt: str, user_message: str, context: str) -> str:
    """Générer une réponse (à adapter selon votre API)"""
    # Ici vous pouvez intégrer votre API préférée (OpenAI, Mistral, etc.)
    # Pour l'exemple, une réponse simple basée sur le contexte
    
    if context.strip():
        return f"Basé sur les documents fournis, voici ma réponse à votre question '{user_message}':\n\n{context[:500]}..."
    else:
        return "Je n'ai pas trouvé d'informations pertinentes dans les documents uploadés pour répondre à votre question."

@app.get("/documents")
async def list_documents():
    """Lister les documents dans la base"""
    try:
        # Récupérer tous les documents
        results = collection.get()
        
        # Grouper par nom de fichier
        files = {}
        for i, metadata in enumerate(results['metadatas']):
            filename = metadata['filename']
            if filename not in files:
                files[filename] = {
                    'filename': filename,
                    'chunks_count': 0
                }
            files[filename]['chunks_count'] += 1
        
        return {"documents": list(files.values())}
        
    except Exception as e:
        logger.error(f"Erreur liste documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents")
async def clear_documents():
    """Vider la base de documents"""
    try:
        # Supprimer la collection et la recréer
        chroma_client.delete_collection("documents")
        global collection
        collection = chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        return {"message": "Base de documents vidée avec succès"}
        
    except Exception as e:
        logger.error(f"Erreur suppression: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Vérification de santé de l'API"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )