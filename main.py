from rag_backend import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_backend:app", host="0.0.0.0", port=8000, reload=True)
