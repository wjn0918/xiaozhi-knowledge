from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import List

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
TEXTS_PATH = DATA_DIR / "faiss_texts.npy"

# 支持通过环境变量覆盖模型目录
DEFAULT_MODEL_DIR = (
    BASE_DIR
    / "models"
    / "sentence-transformers"
    / "all-MiniLM-L6-v2"
)
EMBEDDING_MODEL_DIR = os.getenv("EMBEDDING_MODEL_DIR", str(DEFAULT_MODEL_DIR))


class KnowledgeCreate(BaseModel):
    text: str = Field(..., min_length=1, description="知识文本")


class KnowledgeUpdate(BaseModel):
    text: str = Field(..., min_length=1, description="更新后的知识文本")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)


class KnowledgeItem(BaseModel):
    id: int
    text: str


class VectorStore:
    def __init__(self, index_path: Path = FAISS_INDEX_PATH, texts_path: Path = TEXTS_PATH):
        self.model = SentenceTransformer(EMBEDDING_MODEL_DIR)
        self.dimension = 384
        self.index_path = Path(index_path)
        self.texts_path = Path(texts_path)

        if self.index_path.exists() and self.texts_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.texts: List[str] = np.load(self.texts_path, allow_pickle=True).tolist()
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.texts = []

    def save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        np.save(self.texts_path, np.array(self.texts, dtype=object))

    def _rebuild_index(self) -> None:
        self.index = faiss.IndexFlatL2(self.dimension)
        if self.texts:
            embeddings = self.model.encode(self.texts)
            self.index.add(np.array(embeddings).astype("float32"))
        self.save()

    def list_items(self) -> List[KnowledgeItem]:
        return [KnowledgeItem(id=i, text=text) for i, text in enumerate(self.texts)]

    def add(self, text: str) -> KnowledgeItem:
        embedding = self.model.encode([text])
        self.index.add(np.array(embedding).astype("float32"))
        self.texts.append(text)
        self.save()
        return KnowledgeItem(id=len(self.texts) - 1, text=text)

    def get(self, item_id: int) -> KnowledgeItem:
        if not (0 <= item_id < len(self.texts)):
            raise IndexError("知识不存在")
        return KnowledgeItem(id=item_id, text=self.texts[item_id])

    def update(self, item_id: int, text: str) -> KnowledgeItem:
        if not (0 <= item_id < len(self.texts)):
            raise IndexError("知识不存在")
        self.texts[item_id] = text
        self._rebuild_index()
        return KnowledgeItem(id=item_id, text=text)

    def delete(self, item_id: int) -> None:
        if not (0 <= item_id < len(self.texts)):
            raise IndexError("知识不存在")
        del self.texts[item_id]
        self._rebuild_index()

    def search(self, query: str, k: int = 3) -> List[KnowledgeItem]:
        if not self.texts:
            return []

        embedding = self.model.encode([query])
        _, neighbors = self.index.search(np.array(embedding).astype("float32"), k)
        result = []
        for idx in neighbors[0]:
            if idx < len(self.texts):
                result.append(KnowledgeItem(id=int(idx), text=self.texts[int(idx)]))
        return result


app = FastAPI(title="Knowledge Base API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = VectorStore()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/knowledge", response_model=List[KnowledgeItem])
def list_knowledge() -> List[KnowledgeItem]:
    return store.list_items()


@app.post("/knowledge", response_model=KnowledgeItem)
def create_knowledge(payload: KnowledgeCreate) -> KnowledgeItem:
    return store.add(payload.text)


@app.get("/knowledge/{item_id}", response_model=KnowledgeItem)
def get_knowledge(item_id: int) -> KnowledgeItem:
    try:
        return store.get(item_id)
    except IndexError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err


@app.put("/knowledge/{item_id}", response_model=KnowledgeItem)
def update_knowledge(item_id: int, payload: KnowledgeUpdate) -> KnowledgeItem:
    try:
        return store.update(item_id, payload.text)
    except IndexError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err


@app.delete("/knowledge/{item_id}")
def delete_knowledge(item_id: int) -> dict:
    try:
        store.delete(item_id)
        return {"message": "删除成功"}
    except IndexError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err


@app.post("/knowledge/search", response_model=List[KnowledgeItem])
def search_knowledge(payload: SearchRequest) -> List[KnowledgeItem]:
    return store.search(payload.query, payload.top_k)


@app.get("/faiss/download")
def download_faiss() -> FileResponse:
    if not FAISS_INDEX_PATH.exists() or not TEXTS_PATH.exists():
        raise HTTPException(status_code=404, detail="faiss 文件不存在")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        zip_path = Path(tmp.name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(FAISS_INDEX_PATH, arcname="faiss.index")
        zipf.write(TEXTS_PATH, arcname="faiss_texts.npy")
        metadata = {"total_items": len(store.texts)}
        zipf.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))

    return FileResponse(zip_path, filename="faiss_bundle.zip", media_type="application/zip")
