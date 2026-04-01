import { useEffect, useState } from 'react';

const API_BASE = 'http://127.0.0.1:8000';

export default function App() {
  const [items, setItems] = useState([]);
  const [newText, setNewText] = useState('');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [editingId, setEditingId] = useState(null);
  const [editingText, setEditingText] = useState('');

  const fetchItems = async () => {
    const res = await fetch(`${API_BASE}/knowledge`);
    const data = await res.json();
    setItems(data);
  };

  useEffect(() => {
    fetchItems();
  }, []);

  const addItem = async () => {
    if (!newText.trim()) return;
    await fetch(`${API_BASE}/knowledge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: newText })
    });
    setNewText('');
    fetchItems();
  };

  const removeItem = async (id) => {
    await fetch(`${API_BASE}/knowledge/${id}`, { method: 'DELETE' });
    fetchItems();
  };

  const startEdit = (item) => {
    setEditingId(item.id);
    setEditingText(item.text);
  };

  const saveEdit = async () => {
    if (editingId === null || !editingText.trim()) return;
    await fetch(`${API_BASE}/knowledge/${editingId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: editingText })
    });
    setEditingId(null);
    setEditingText('');
    fetchItems();
  };

  const search = async () => {
    if (!query.trim()) {
      setResults([]);
      return;
    }
    const res = await fetch(`${API_BASE}/knowledge/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: 5 })
    });
    const data = await res.json();
    setResults(data);
  };

  return (
    <main className="container">
      <h1>知识库管理</h1>

      <section className="card">
        <h2>新增知识</h2>
        <div className="row">
          <input
            value={newText}
            onChange={(e) => setNewText(e.target.value)}
            placeholder="输入知识内容"
          />
          <button onClick={addItem}>新增</button>
        </div>
      </section>

      <section className="card">
        <h2>知识列表（支持增删改查）</h2>
        {items.map((item) => (
          <div className="item" key={item.id}>
            {editingId === item.id ? (
              <>
                <input value={editingText} onChange={(e) => setEditingText(e.target.value)} />
                <button onClick={saveEdit}>保存</button>
              </>
            ) : (
              <>
                <span>{item.id}. {item.text}</span>
                <div className="actions">
                  <button onClick={() => startEdit(item)}>编辑</button>
                  <button className="danger" onClick={() => removeItem(item.id)}>删除</button>
                </div>
              </>
            )}
          </div>
        ))}
      </section>

      <section className="card">
        <h2>语义搜索</h2>
        <div className="row">
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="输入查询问题" />
          <button onClick={search}>搜索</button>
        </div>
        {results.map((item) => (
          <p key={item.id}>#{item.id} - {item.text}</p>
        ))}
      </section>

      <section className="card">
        <h2>下载 FAISS 数据</h2>
        <a className="button" href={`${API_BASE}/faiss/download`}>下载 faiss_bundle.zip</a>
      </section>
    </main>
  );
}
