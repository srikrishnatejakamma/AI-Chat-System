import { useEffect, useRef, useState } from 'react';
import { FileText, LoaderCircle, Monitor, Moon, RefreshCw, Search, ShieldCheck, Sun, Trash2, Upload, X } from 'lucide-react';
import { IconButton } from './Response.jsx';
import { hasModel, modelName } from './api.js';

export function Modal({ title, children, onClose, className = '' }) {
  const ref = useRef(null);
  useEffect(() => { const dialog = ref.current; dialog.showModal(); return () => dialog.close(); }, []);
  return <dialog ref={ref} className={`modal ${className}`} aria-label={title} onCancel={onClose} onClick={event => { if (event.target === event.currentTarget) onClose(); }}>
    <div className="modal-heading"><h2>{title}</h2><IconButton label="Close dialog" onClick={onClose}><X size={19} /></IconButton></div>{children}
  </dialog>;
}

export function Library({ documents, onClose, onUpload, onUse, onDelete, busy, uploading }) {
  const [search, setSearch] = useState('');
  return <Modal title="My documents" onClose={onClose} className="library-modal"><p className="modal-description">Your documents, ready for a conversation. Stored on this device.</p>
    <div className="library-top"><div className="search-field"><Search size={16} /><input aria-label="Search documents" placeholder="Search documents" value={search} onChange={event => setSearch(event.target.value)} /></div>
      <button className="primary-button" onClick={onUpload} disabled={busy || uploading}>{uploading ? <LoaderCircle size={16} className="spin" /> : <Upload size={16} />}Upload</button></div>
    <div className="document-list">{documents.filter(doc => doc.name.toLowerCase().includes(search.toLowerCase())).map(doc => <div className="document-row" key={doc.id}><div className="file-icon"><FileText size={22} /></div><div className="document-info"><strong>{doc.name}</strong><span>{doc.chunks} passages · {(doc.bytes / 1024).toFixed(1)} KB</span></div>
      <button className="secondary-button" onClick={() => onUse(doc)} disabled={busy}>Use in chat</button><IconButton label={`Remove ${doc.name}`} disabled={busy || uploading} onClick={() => onDelete(doc)}><Trash2 size={16} /></IconButton></div>)}
      {documents.length === 0 && <button className="empty-library" onClick={onUpload} disabled={uploading}><Upload size={28} /><strong>Add your first document</strong><span>PDF, DOCX, or TXT · Up to 25 MB</span></button>}
      {documents.length > 0 && !documents.some(doc => doc.name.toLowerCase().includes(search.toLowerCase())) && <p className="empty-filter">No documents match your search.</p>}
    </div><div className="modal-footnote"><ShieldCheck size={15} />Files are parsed locally. Indexed text stays on this device.</div>
  </Modal>;
}

export function Settings({ status, model, setModel, topK, setTopK, theme, setTheme, onRefresh, onClose, busy }) {
  const [refreshing, setRefreshing] = useState(false);
  async function refresh() { setRefreshing(true); try { await onRefresh(); } finally { setRefreshing(false); } }
  return <Modal title="Settings" onClose={onClose} className="settings-modal"><div className="setting-row"><div><h3>Appearance</h3><p>Make this space your own.</p></div><div className="theme-options">{[['light', Sun], ['dark', Moon], ['system', Monitor]].map(([value, Icon]) => <button key={value} className={theme === value ? 'selected' : ''} aria-label={`${value} theme`} aria-pressed={theme === value} title={value} onClick={() => setTheme(value)}><Icon size={18} /></button>)}</div></div>
    <div className="setting-row"><div><h3>Chat model</h3><p>Runs with Ollama on your computer.</p></div><select aria-label="Default chat model" value={model} onChange={event => setModel(event.target.value)} disabled={busy}>{status?.models?.map(value => <option key={value} value={value}>{modelName(value)}{hasModel(status, value) ? '' : ' (not installed)'}</option>)}</select></div>
    <div className="setting-row"><div><h3>Retrieved passages</h3><p>How much evidence to give each answer.</p></div><div className="range-setting"><input aria-label="Retrieved passages" type="range" min="1" max="6" value={topK} disabled={busy} onChange={event => setTopK(Number(event.target.value))} /><span>{topK}</span></div></div>
    <div className="connection-card"><div className="connection-heading"><span className={`status-dot ${status?.online ? 'online' : ''}`} /><strong>{status?.online ? 'Ollama connected' : 'Ollama is offline'}</strong><IconButton label="Refresh connection" disabled={refreshing} onClick={refresh}><RefreshCw size={16} className={refreshing ? 'spin' : ''} /></IconButton></div>
      <p>Embedding model: {status?.embedding_model || 'nomic-embed-text'}</p>{(!status?.online || !status?.embedding_ready || !hasModel(status, model)) && <div className="setup-instructions"><p>Start local Ollama and download the models to begin.</p><code>ollama pull {status?.embedding_model || 'nomic-embed-text'}<br />ollama pull {model || 'phi3'}</code><p>Using Docker? Prefix each command with <code>docker compose exec ollama</code>.</p></div>}
    </div><div className="modal-footnote"><ShieldCheck size={16} />Your conversations and documents stay local.</div>
  </Modal>;
}

export function Confirm({ title, description, actionLabel = 'Delete', initialValue, onConfirm, onClose }) {
  const [value, setValue] = useState(initialValue || '');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  return <Modal title={title} onClose={onClose} className="confirm-modal"><p className="modal-description">{description}</p>
    {initialValue !== undefined && <input className="rename-input" aria-label="Conversation name" value={value} onChange={event => setValue(event.target.value)} maxLength={120} autoFocus />}
    {error && <p className="error-text" role="alert">{error}</p>}
    <div className="dialog-actions"><button className="secondary-button" onClick={onClose} disabled={busy}>Cancel</button><button className={initialValue === undefined ? 'danger-button' : 'primary-button'} disabled={busy || (initialValue !== undefined && !value.trim())} onClick={async () => { setBusy(true); setError(''); try { await onConfirm(value); onClose(); } catch (error) { setError(error.message); } finally { setBusy(false); } }}>{busy && <LoaderCircle size={15} className="spin" />}{actionLabel}</button></div>
  </Modal>;
}

export function Sources({ sources, selected, onClose }) {
  const selectedRef = useRef(null);
  useEffect(() => { selectedRef.current?.scrollIntoView({ block: 'nearest' }); }, [selected]);
  return <><button className="sources-backdrop" aria-label="Close sources" onClick={onClose} /><aside className="sources-panel" aria-label="Retrieved sources"><div className="modal-heading"><div><h2>Sources</h2><p>{sources.length} retrieved passages</p></div><IconButton label="Close sources" onClick={onClose}><X size={19} /></IconButton></div><p className="sources-explanation">These excerpts were supplied to the model for this answer.</p><div className="source-list">{sources.map(source => <section key={source.number} ref={selected === source.number ? selectedRef : null} className={`source-card ${selected === source.number ? 'selected' : ''}`}><header><span>{source.number}</span><FileText size={16} /><strong>{source.name}</strong></header>{source.page && <small>Page {source.page}</small>}<p>{source.text}</p></section>)}</div></aside></>;
}
