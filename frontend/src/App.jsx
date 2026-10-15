import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { ArrowDown, BookOpen, ChevronDown, Columns2, Download, FileText, FolderClosed, LoaderCircle, PanelLeft, Search, ShieldCheck, Sparkles, X } from 'lucide-react';
import { downloadChat, hasModel, modelName, request } from './api.js';
import Composer from './Composer.jsx';
import { Confirm, Library, Settings, Sources } from './Dialogs.jsx';
import Response, { IconButton, Logo } from './Response.jsx';
import Sidebar from './Sidebar.jsx';

function preferences() {
  try { return JSON.parse(localStorage.getItem('local-chat-preferences') || '{}'); }
  catch { return {}; }
}

const suggestions = [
  { icon: FileText, label: 'Summarize a document', prompt: 'Summarize the key points in my documents.' },
  { icon: Search, label: 'Find a specific detail', prompt: 'What does the document say about ' },
  { icon: Columns2, label: 'Compare key points', prompt: 'Compare the main ideas across my documents.' },
  { icon: Sparkles, label: 'Explain something', prompt: 'Explain the main concepts in my documents in simple terms.' },
];

export default function App() {
  const [prefs] = useState(preferences);
  const [theme, setTheme] = useState(prefs.theme || 'light');
  const [model, setModel] = useState(prefs.model || '');
  const [topK, setTopK] = useState(prefs.topK || 3);
  const [status, setStatus] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [chats, setChats] = useState([]);
  const [chat, setChat] = useState(null);
  const [booting, setBooting] = useState(true);
  const [draft, setDraft] = useState('');
  const [scope, setScope] = useState('all');
  const [attachments, setAttachments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [requesting, setRequesting] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth >= 900);
  const [modal, setModal] = useState(null);
  const [confirm, setConfirm] = useState(null);
  const [sources, setSources] = useState(null);
  const [toast, setToast] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [atBottom, setAtBottom] = useState(true);
  const inputRef = useRef(null);
  const filesRef = useRef(null);
  const scrollRef = useRef(null);
  const nearBottom = useRef(true);
  const dragDepth = useRef(0);
  const submitLock = useRef(false);
  const generating = Boolean(chat?.active);
  const busy = generating || requesting;
  const ready = status?.embedding_ready && hasModel(status, model);
  const messages = chat?.messages || [];

  const notify = useCallback((message, kind = 'error') => setToast({ message, kind, id: Date.now() }), []);
  const refreshChats = useCallback(async () => setChats(await request('/chats')), []);
  const refreshDocuments = useCallback(async () => setDocuments(await request('/documents')), []);
  const refreshStatus = useCallback(async () => {
    try { setStatus(await request('/status?refresh=true')); }
    catch (error) { notify(error.message); }
  }, [notify]);

  function applyChat(value) {
    setChat(value);
    try { if (value) localStorage.setItem('local-chat-last', value.id); else localStorage.removeItem('local-chat-last'); } catch { /* Optional browser convenience. */ }
  }

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const [connection, docs, history] = await Promise.all(['/status', '/documents', '/chats'].map(path => request(path, { signal: controller.signal })));
        if (controller.signal.aborted) return;
        setStatus(connection); setDocuments(docs); setChats(history);
        setModel(previous => connection.models.includes(previous) ? previous : connection.default_model);
        let last;
        try { last = localStorage.getItem('local-chat-last'); } catch { /* No browser storage. */ }
        const target = history.find(item => item.id === last);
        if (target) {
          const saved = await request(`/chats/${target.id}`, { signal: controller.signal });
          if (!controller.signal.aborted) setChat(saved);
        }
      } catch (error) { if (error.name !== 'AbortError') notify(`Could not load the local workspace. ${error.message}`); }
      finally { if (!controller.signal.aborted) setBooting(false); }
    }
    load();
    return () => controller.abort();
  }, [notify]);

  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const apply = () => document.documentElement.dataset.theme = theme === 'system' ? media.matches ? 'dark' : 'light' : theme;
    apply(); media.addEventListener('change', apply);
    try { localStorage.setItem('local-chat-preferences', JSON.stringify({ theme, model, topK })); } catch { /* Preferences are optional. */ }
    return () => media.removeEventListener('change', apply);
  }, [theme, model, topK]);

  useEffect(() => {
    if (!chat?.active) return;
    let disposed = false;
    let timer;
    const controller = new AbortController();
    async function poll() {
      try {
        const next = await request(`/chats/${chat.id}`, { signal: controller.signal });
        if (disposed) return;
        setChat(next);
        if (!next.active) { await refreshChats(); return; }
      } catch (error) {
        if (disposed || error.name === 'AbortError') return;
        notify('Connection interrupted. Reconnecting to your response…');
      }
      if (!disposed) timer = setTimeout(poll, 350);
    }
    timer = setTimeout(poll, 200);
    return () => { disposed = true; clearTimeout(timer); controller.abort(); };
  }, [chat?.id, chat?.active, notify, refreshChats]);

  useEffect(() => {
    const timer = setInterval(() => request('/status').then(setStatus).catch(() => {}), 15000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), 6500);
    return () => clearTimeout(timer);
  }, [toast]);

  useLayoutEffect(() => {
    if (scrollRef.current && nearBottom.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [chat?.id, messages.length, messages.at(-1)?.content]);

  useEffect(() => {
    const close = event => { if (event.key === 'Escape') { setSources(null); setDragging(false); } };
    window.addEventListener('keydown', close);
    return () => window.removeEventListener('keydown', close);
  }, []);

  async function action(callback) {
    if (submitLock.current) return;
    submitLock.current = true; setRequesting(true);
    try { await callback(); }
    catch (error) { notify(error.message); }
    finally { submitLock.current = false; setRequesting(false); }
  }

  async function stop() {
    if (!chat?.active) return;
    await action(async () => { applyChat(await request(`/chats/${chat.id}/stop`, { method: 'POST' })); await refreshChats(); });
  }

  async function newChat() {
    await action(async () => {
      if (chat?.active) await request(`/chats/${chat.id}/stop`, { method: 'POST' });
      applyChat(null); setDraft(''); setAttachments([]); setScope('all'); setSources(null);
      nearBottom.current = true;
      if (window.innerWidth < 900) setSidebarOpen(false);
      await refreshChats();
      setTimeout(() => inputRef.current?.focus(), 0);
    });
  }

  useEffect(() => {
    const shortcut = event => { if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); newChat(); } };
    window.addEventListener('keydown', shortcut);
    return () => window.removeEventListener('keydown', shortcut);
  });

  async function selectChat(id) {
    if (id === chat?.id) { if (window.innerWidth < 900) setSidebarOpen(false); return; }
    await action(async () => {
      if (chat?.active) await request(`/chats/${chat.id}/stop`, { method: 'POST' });
      applyChat(await request(`/chats/${id}`)); setDraft(''); setSources(null); setAttachments([]); setScope('all');
      nearBottom.current = true;
      if (window.innerWidth < 900) setSidebarOpen(false);
      await refreshChats();
    });
  }

  async function send() {
    if (!draft.trim() || busy || uploading || !ready) return;
    await action(async () => {
      const target = chat || await request('/chats', { method: 'POST' });
      if (!chat) applyChat(target);
      const documentIds = scope === 'all' ? null : scope === 'attachments'
        ? attachments.filter(file => file.document_id && !file.error).map(file => file.document_id) : [scope];
      const next = await request(`/chats/${target.id}/messages`, { method: 'POST', body: { content: draft.trim(), model, document_ids: documentIds, top_k: topK } });
      applyChat(next); setDraft(''); setAttachments([]); if (scope === 'attachments') setScope('all');
      nearBottom.current = true;
      await refreshChats();
    });
  }

  async function regenerate(messageId) {
    await action(async () => {
      setSources(null); nearBottom.current = true;
      applyChat(await request(`/chats/${chat.id}/responses/${messageId}/regenerate`, { method: 'POST', body: { model } }));
      await refreshChats();
    });
  }

  async function chooseVersion(messageId, index) {
    await action(async () => {
      setSources(null);
      applyChat(await request(`/chats/${chat.id}/responses/${messageId}/version`, { method: 'POST', body: { index } }));
    });
  }

  async function uploadFiles(fileList) {
    if (uploading || busy) { notify('Stop the current response before adding documents.'); return; }
    const files = Array.from(fileList);
    if (!files.length) return;
    setUploading(true); setScope('attachments');
    for (const file of files) {
      const key = crypto.randomUUID();
      if (!/\.(pdf|docx|txt)$/i.test(file.name) || file.size > 25 * 1024 * 1024) {
        notify(`${file.name}: choose a PDF, DOCX, or TXT file up to 25 MB.`); continue;
      }
      setAttachments(previous => [...previous, { key, name: file.name, indexing: true }]);
      try {
        const body = new FormData(); body.append('file', file);
        const result = await request('/documents', { method: 'POST', body });
        setAttachments(previous => previous.map(item => item.key === key ? { ...item, indexing: false, document_id: result.document_id } : item));
        await refreshDocuments();
        if (result.duplicate) notify(`${result.name} is already in your library.`, 'success');
      } catch (error) {
        setAttachments(previous => previous.map(item => item.key === key ? { ...item, indexing: false, error: true } : item));
        notify(error.message);
      }
    }
    setUploading(false);
    setTimeout(() => inputRef.current?.focus(), 0);
  }

  function removeAttachment(key) {
    setAttachments(previous => { const next = previous.filter(file => file.key !== key); if (!next.length && scope === 'attachments') setScope('all'); return next; });
  }

  function deleteDocument(doc) {
    setConfirm({ title: 'Remove document?', description: `Remove “${doc.name}” and its indexed passages? Saved chats that cite it will also be removed.`,
      onConfirm: async () => {
        const result = await request(`/documents/${doc.id}`, { method: 'DELETE' });
        if (result.deleted_chat_ids.includes(chat?.id)) applyChat(null);
        if (scope === doc.id) setScope('all');
        setAttachments(previous => previous.filter(file => file.document_id !== doc.id));
        setSources(null); await Promise.all([refreshDocuments(), refreshChats()]);
      } });
  }

  const composer = <Composer draft={draft} setDraft={setDraft} onSend={send} onStop={stop} generating={generating}
    disabled={booting || requesting || uploading || !ready || !documents.length || (scope === 'attachments' && !attachments.some(file => file.document_id && !file.error))}
    uploading={uploading} documents={documents} scope={scope} setScope={setScope} onUpload={() => filesRef.current?.click()}
    onLibrary={() => setModal('library')} attachments={attachments} onRemoveAttachment={removeAttachment} inputRef={inputRef} />;

  return <div className={`app-shell ${sidebarOpen ? 'sidebar-is-open' : ''}`} onDragEnter={event => {
    if (event.dataTransfer.types.includes('Files')) { event.preventDefault(); dragDepth.current++; setDragging(true); }
  }} onDragLeave={event => { if (event.dataTransfer.types.includes('Files')) { dragDepth.current--; if (dragDepth.current <= 0) setDragging(false); } }}
    onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); setDragging(false); dragDepth.current = 0; if (event.dataTransfer.files.length) uploadFiles(event.dataTransfer.files); }}>
    <Sidebar open={sidebarOpen} onToggle={() => setSidebarOpen(value => !value)} chats={chats} chatId={chat?.id} online={status?.online}
      onNew={newChat} onSelect={selectChat} onLibrary={() => setModal('library')} onSettings={() => setModal('settings')}
      onRename={target => setConfirm({ title: 'Rename chat', description: 'Give this conversation a name.', actionLabel: 'Save', initialValue: target.title,
        onConfirm: async title => { const saved = await request(`/chats/${target.id}`, { method: 'PATCH', body: { title } }); if (chat?.id === target.id) applyChat(saved); await refreshChats(); } })}
      onDelete={target => setConfirm({ title: 'Delete chat?', description: `“${target.title}” and all its response versions will be permanently deleted.`,
        onConfirm: async () => { await request(`/chats/${target.id}`, { method: 'DELETE' }); if (chat?.id === target.id) applyChat(null); setSources(null); await refreshChats(); } })} />
    <main className="main-pane"><header className="main-header"><div className="header-left">{!sidebarOpen && <IconButton label="Open sidebar" onClick={() => setSidebarOpen(true)}><PanelLeft size={20} /></IconButton>}
      <span className="app-name">Local Chat</span><div className="model-control"><select aria-label="Chat model" disabled={busy} value={model} onChange={event => setModel(event.target.value)}>{status?.models?.map(value => <option key={value} value={value}>{modelName(value)}</option>)}</select><ChevronDown size={14} /></div></div>
      <div className="header-actions"><button className={`privacy-badge ${status?.online ? '' : 'offline'}`} onClick={() => setModal('settings')}><ShieldCheck size={14} /><span>{status?.online ? 'Local only' : 'Connect Ollama'}</span></button><IconButton label="Document library" onClick={() => setModal('library')}><FolderClosed size={18} /></IconButton>{chat && <IconButton label="Download conversation" onClick={() => downloadChat(chat).catch(error => notify(error.message))}><Download size={18} /></IconButton>}</div>
    </header>
    {booting ? <div className="workspace-loading" role="status"><LoaderCircle size={25} className="spin" /><span>Opening your workspace…</span></div> : <div className="workspace-body">
      {messages.length ? <><div className="conversation-scroll" ref={scrollRef} onScroll={event => { const element = event.currentTarget; nearBottom.current = element.scrollHeight - element.scrollTop - element.clientHeight < 100; setAtBottom(nearBottom.current); }}><div className="conversation-content">
        {messages.map(message => message.role === 'user' ? <div className="user-message" key={message.id}><div>{message.content}</div></div> : <Response key={message.id} message={message} busy={busy} onRegenerate={regenerate} onVersion={chooseVersion} onSources={(items, selected) => setSources({ items, selected })} notify={notify} />)}
        <div className="conversation-end" />
      </div></div>{!atBottom && <button className="scroll-bottom icon-button" aria-label="Scroll to latest response" onClick={() => { nearBottom.current = true; scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' }); }}><ArrowDown size={19} /></button>}<div className="composer-dock">{composer}</div></>
        : <div className="welcome"><div className="welcome-intro"><div className="welcome-mark"><Logo size={35} /></div><h1>What can I help with?</h1><p>A conversation with your documents. Entirely on your device.</p></div><div className="welcome-composer">{composer}</div>
          <div className="suggestions">{suggestions.map(({ icon: Icon, label, prompt }) => <button key={label} onClick={() => { setDraft(prompt); inputRef.current?.focus(); }}><Icon size={16} /><span>{label}</span></button>)}</div>
          {!documents.length && <button className="first-document" onClick={() => filesRef.current?.click()}><BookOpen size={15} />Add your first document to get started</button>}
        </div>}
      <footer className="main-footer">Local Chat can make mistakes. Check the sources.</footer>
    </div>}
    </main>
    <input className="sr-only" ref={filesRef} type="file" accept=".pdf,.docx,.txt" multiple aria-label="Upload documents" onChange={event => { uploadFiles(event.target.files); event.target.value = ''; }} />
    {modal === 'library' && <Library documents={documents} busy={busy} uploading={uploading} onClose={() => setModal(null)} onUpload={() => filesRef.current?.click()} onDelete={deleteDocument} onUse={doc => { setScope(doc.id); setAttachments([]); setModal(null); inputRef.current?.focus(); }} />}
    {modal === 'settings' && <Settings status={status} model={model} setModel={setModel} topK={topK} setTopK={setTopK} theme={theme} setTheme={setTheme} busy={busy} onRefresh={refreshStatus} onClose={() => setModal(null)} />}
    {confirm && <Confirm {...confirm} onClose={() => setConfirm(null)} />}
    {sources && <Sources sources={sources.items} selected={sources.selected} onClose={() => setSources(null)} />}
    {toast && <div className={`toast ${toast.kind}`} role={toast.kind === 'error' ? 'alert' : 'status'}><span>{toast.message}</span><IconButton label="Dismiss notification" onClick={() => setToast(null)}><X size={16} /></IconButton></div>}
    {dragging && <div className="drop-overlay"><UploadMark /><h2>Drop your documents here</h2><p>PDF, DOCX, or TXT · Up to 25 MB per file</p></div>}
  </div>;
}

function UploadMark() { return <div className="drop-mark"><FileText size={38} /></div>; }
