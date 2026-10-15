import { useState } from 'react';
import { Ellipsis, FolderClosed, MessageSquare, PanelLeftClose, Search, Settings, ShieldCheck, SquarePen } from 'lucide-react';
import { IconButton, Logo } from './Response.jsx';

export default function Sidebar({ open, onToggle, chats, chatId, onNew, onSelect, onLibrary, onSettings, onRename, onDelete, online }) {
  const [search, setSearch] = useState('');
  const [searching, setSearching] = useState(false);
  const filtered = chats.filter(chat => chat.title.toLowerCase().includes(search.toLowerCase()));
  return <>
    {open && <button className="sidebar-backdrop" aria-label="Close sidebar" onClick={onToggle} />}
    <aside className={`sidebar ${open ? 'open' : 'collapsed'}`} aria-label="Chat sidebar">
      <div className="sidebar-heading"><button className="brand-button" onClick={onNew} aria-label="Local Chat home"><Logo size={29} /><span>Local Chat</span></button>
        <IconButton label={open ? 'Close sidebar' : 'Open sidebar'} onClick={onToggle}><PanelLeftClose size={19} /></IconButton>
      </div>
      <nav className="sidebar-nav">
        <button className="nav-button" onClick={onNew} aria-label="New chat"><SquarePen size={19} /><span>New chat</span><kbd>⌘ K</kbd></button>
        <button className={`nav-button ${searching ? 'selected' : ''}`} aria-label="Search chats" onClick={() => { if (!open) onToggle(); setSearching(value => !value); }}><Search size={19} /><span>Search chats</span></button>
        <button className="nav-button" onClick={onLibrary} aria-label="Document library"><FolderClosed size={19} /><span>My documents</span><span className="nav-local">Local</span></button>
      </nav>
      <div className="chat-history">
        {searching && <div className="history-search"><Search size={15} /><input autoFocus aria-label="Search conversations" placeholder="Search conversations" value={search} onChange={event => setSearch(event.target.value)} /><button onClick={() => { setSearching(false); setSearch(''); }} aria-label="Close search">×</button></div>}
        <h2>Your chats</h2>
        {filtered.length === 0 && <p className="history-empty">{search ? 'No conversations found.' : 'Your conversations will appear here.'}</p>}
        {filtered.map(chat => <div key={chat.id} className={`chat-row ${chatId === chat.id ? 'active' : ''}`}>
          <button className="chat-link" onClick={() => onSelect(chat.id)} title={chat.title}><MessageSquare size={15} /><span>{chat.title}</span>{chat.active && <span className="tiny-spinner" />}</button>
          <details className="chat-menu"><summary aria-label={`Actions for ${chat.title}`}><Ellipsis size={17} /></summary><div className="dropdown-menu">
            <button onClick={event => { event.currentTarget.closest('details').removeAttribute('open'); onRename(chat); }}>Rename</button>
            <button className="danger" onClick={event => { event.currentTarget.closest('details').removeAttribute('open'); onDelete(chat); }}>Delete chat</button>
          </div></details>
        </div>)}
      </div>
      <div className="sidebar-footer"><div className="local-status"><ShieldCheck size={15} /><span>Only on your device</span><span className={`status-dot ${online ? 'online' : ''}`} /></div>
        <button className="workspace-button" onClick={onSettings} aria-label="Settings"><span className="profile-avatar">L</span><span><strong>Personal workspace</strong><small>Local models · Private</small></span><Settings size={17} /></button>
      </div>
    </aside>
  </>;
}
