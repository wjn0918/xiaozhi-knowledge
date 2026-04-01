## 项目说明

本项目提供：

- FastAPI 后端接口（知识库增删改查 + 语义搜索）
- FAISS 索引文件打包下载
- React 前端页面（调用后端接口）

## 1. 启动后端

```bash
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

接口示例：

- `GET /knowledge`：查询全部知识
- `POST /knowledge`：新增知识
- `GET /knowledge/{id}`：查询单条知识
- `PUT /knowledge/{id}`：更新知识
- `DELETE /knowledge/{id}`：删除知识
- `POST /knowledge/search`：语义搜索
- `GET /faiss/download`：下载 `faiss_bundle.zip`

## 2. 启动前端

```bash
cd frontend
npm install
npm run dev
```

浏览器打开 `http://127.0.0.1:5173`。

## 3. 数据文件

运行后会在 `data/` 目录生成：

- `faiss.index`
- `faiss_texts.npy`

下载接口会将以上文件与 metadata 一起打包为 zip。
