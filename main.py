from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

EMBEDDING_MODEL_DIR = r"C:\Users\wjn\yx\xiaozhi-esp32-server\main\xiaozhi-server\models\sentence-transformers\all-MiniLM-L6-v2"
# =========================
# 1️⃣ 向量数据库
# =========================
class VectorStore:
    def __init__(self, index_path="faiss.index"):
        self.model = SentenceTransformer(EMBEDDING_MODEL_DIR)
        self.dimension = 384
        self.index_path = index_path

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.texts = np.load("faiss_texts.npy", allow_pickle=True).tolist()
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.texts = []

    def save(self):
        faiss.write_index(self.index, self.index_path)
        np.save("faiss_texts.npy", np.array(self.texts, dtype=object))

    def add(self, text):
        embedding = self.model.encode([text])
        self.index.add(np.array(embedding).astype("float32"))
        self.texts.append(text)
        self.save()

    def search(self, query, k=3):
        if len(self.texts) == 0:
            return []

        embedding = self.model.encode([query])
        D, I = self.index.search(np.array(embedding).astype("float32"), k)
        return [self.texts[i] for i in I[0] if i < len(self.texts)]

    def rebuild_index(self):
        """重建索引（删除后使用）"""
        self.index = faiss.IndexFlatL2(self.dimension)
        if len(self.texts) > 0:
            embeddings = self.model.encode(self.texts)
            self.index.add(np.array(embeddings).astype("float32"))
        self.save()

    def delete(self, index):
        """删除指定索引的文本"""
        if 0 <= index < len(self.texts):
            del self.texts[index]
            self.rebuild_index()


if __name__ == "__main__":
    store = VectorStore()
    # store.add("这是第一条文本")
    # store.add("这是第二条文本")
    # print(store.search("第一条"))
    # store.delete(0)
    # print(store.search("第一条"))
    print(store.texts)