import { useEffect, useRef, useState } from 'react';
import { ArrowUp, ChevronDown, FileText, FolderOpen, LoaderCircle, Plus, Square, Upload, X } from 'lucide-react';
import { IconButton } from './Response.jsx';

export default function Composer({ draft, setDraft, onSend, onStop, generating, disabled, uploading, documents, scope, setScope, onUpload, onLibrary, attachments, onRemoveAttachment, inputRef }) {
  const [menu, setMenu] = useState(false);
  const menuRef = useRef(null);
  useEffect(() => {
    const close = event => { if (menuRef.current && !menuRef.current.contains(event.target)) setMenu(false); };
    document.addEventListener('pointerdown', close);
    return () => document.removeEventListener('pointerdown', close);
  }, []);
  useEffect(() => {
    const input = inputRef.current;
    if (input) { input.style.height = 'auto'; input.style.height = `${Math.min(input.scrollHeight, 180)}px`; }
  }, [draft, inputRef]);

  return <form className={`composer ${generating ? 'is-generating' : ''}`} onSubmit={event => { event.preventDefault(); if (!disabled && draft.trim() && !generating) onSend(); }}>
    {attachments.length > 0 && <div className="attachments">{attachments.map(file => <div className={`attachment ${file.error ? 'attachment-error' : ''}`} key={file.key}>
      {file.indexing ? <LoaderCircle size={18} className="spin" /> : <FileText size={18} />}<span><strong>{file.name}</strong><small>{file.indexing ? 'Indexing locally…' : file.error ? 'Could not index' : 'Ready to use'}</small></span>
      <IconButton label={`Remove attachment ${file.name}`} onClick={() => onRemoveAttachment(file.key)} disabled={file.indexing}><X size={13} /></IconButton>
    </div>)}</div>}
    <label className="sr-only" htmlFor="message-input">Message Local Chat</label>
    <textarea id="message-input" ref={inputRef} rows={1} value={draft} maxLength={4000} placeholder={documents.length ? 'Ask anything about your documents' : 'Add a document to get started'}
      onChange={event => setDraft(event.target.value)} onKeyDown={event => {
        if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
          event.preventDefault();
          if (!disabled && draft.trim() && !generating) onSend();
        }
      }} />
    <div className="composer-bottom"><div className="composer-tools"><div ref={menuRef} className="attachment-menu-wrap">
      <IconButton label="Add documents" className="attach-button" aria-expanded={menu} disabled={uploading || generating} onClick={() => setMenu(value => !value)}><Plus size={22} /></IconButton>
      {menu && <div className="dropdown-menu attachment-menu"><button onClick={() => { setMenu(false); onUpload(); }}><Upload size={16} />Upload a document</button><button onClick={() => { setMenu(false); onLibrary(); }}><FolderOpen size={16} />Choose from library</button><small>PDF, DOCX, TXT · Up to 25 MB</small></div>}
    </div>
      <div className="scope-control"><FileText size={14} /><select aria-label="Search in" value={scope} disabled={generating || uploading} onChange={event => setScope(event.target.value)}>
        <option value="all">All documents</option>{attachments.length > 0 && <option value="attachments">Attached files ({attachments.length})</option>}{documents.map(doc => <option key={doc.id} value={doc.id}>{doc.name}</option>)}
      </select><ChevronDown size={12} /></div>
    </div><div className="send-tools">{draft.length > 3800 && <span className="character-count">{draft.length}/4000</span>}
      {generating ? <button type="button" className="send-button stop-button" aria-label="Stop response" title="Stop response" onClick={onStop}><Square size={14} fill="currentColor" /></button>
        : <button className="send-button" type="submit" aria-label="Send message" title="Send message" disabled={disabled || !draft.trim()}>{uploading ? <LoaderCircle size={19} className="spin" /> : <ArrowUp size={20} strokeWidth={2.5} />}</button>}
    </div></div>
  </form>;
}
