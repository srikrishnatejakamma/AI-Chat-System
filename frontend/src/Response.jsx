import { useState } from 'react';
import { BookOpen, Check, ChevronLeft, ChevronRight, Copy, Ellipsis, RotateCcw } from 'lucide-react';
import Markdown from './Markdown.jsx';
import { modelName } from './api.js';

export function IconButton({ label, children, className = '', ...props }) {
  return <button className={`icon-button ${className}`} type="button" aria-label={label} title={label} {...props}>{children}</button>;
}

export function Logo({ size = 24 }) {
  return <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
    <path d="M16 3.5 26.8 9.75v12.5L16 28.5 5.2 22.25V9.75L16 3.5Z" stroke="currentColor" strokeWidth="2.1" strokeLinejoin="round" />
    <path d="m16 3.5 0 12.5M26.8 9.75 16 16 5.2 9.75M5.2 22.25 16 16l10.8 6.25M16 16v12.5" stroke="currentColor" strokeWidth="2.1" strokeLinejoin="round" />
  </svg>;
}

export default function Response({ message, busy, onRegenerate, onVersion, onSources, notify }) {
  const [copied, setCopied] = useState(false);
  const [details, setDetails] = useState(false);
  const version = message.versions[message.selected_version];
  const active = ['retrieving', 'generating'].includes(version.status);
  const retry = ['failed', 'stopped'].includes(version.status);
  const sources = version.sources || [];

  async function copy() {
    try {
      await navigator.clipboard.writeText(version.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch { notify('Could not copy. Select the text and copy it manually.'); }
  }

  return <article className="assistant-message" aria-label="Assistant response">
    <div className="assistant-avatar"><Logo size={23} /></div>
    <div className="assistant-body">
      {version.content && <Markdown content={version.content} sources={sources} onSource={number => onSources(sources, number)} />}
      {active && <div className="thinking" role="status"><span className="thinking-dot" />{version.status === 'retrieving' ? 'Searching your documents' : version.content ? 'Writing' : 'Thinking'}</div>}
      {version.status === 'stopped' && <p className="response-note">Response stopped{version.content ? '' : ' before any text was received'}.</p>}
      {version.status === 'failed' && <p className="response-note error-text">Couldn’t finish this response. You can retry.</p>}
      {sources.length > 0 && !active && <button className="sources-pill" onClick={() => onSources(sources)}><BookOpen size={14} /><span>Sources</span><span className="source-count">{sources.length}</span></button>}
      <div className="response-toolbar">
        {message.versions.length > 1 && <div className="version-navigation" aria-label="Response versions">
          <IconButton label="Previous response" disabled={busy || message.selected_version === 0} onClick={() => onVersion(message.id, message.selected_version - 1)}><ChevronLeft size={17} /></IconButton>
          <span className="version-count" aria-label={`Response version ${message.selected_version + 1} of ${message.versions.length}`}>{message.selected_version + 1}/{message.versions.length}</span>
          <IconButton label="Next response" disabled={busy || message.selected_version === message.versions.length - 1} onClick={() => onVersion(message.id, message.selected_version + 1)}><ChevronRight size={17} /></IconButton>
        </div>}
        <IconButton label={copied ? 'Copied' : 'Copy response'} disabled={!version.content} onClick={copy}>{copied ? <Check size={16} /> : <Copy size={16} />}</IconButton>
        {!active && <IconButton label={retry ? 'Retry response' : 'Regenerate response'} disabled={busy} onClick={() => onRegenerate(message.id)}><RotateCcw size={17} /></IconButton>}
        <div className="response-details-wrap">
          <IconButton label="Response details" aria-expanded={details} onClick={() => setDetails(value => !value)}><Ellipsis size={18} /></IconButton>
          {details && <div className="response-details"><strong>{modelName(version.model)}</strong><span>{version.status[0].toUpperCase() + version.status.slice(1)}</span>
            <time dateTime={version.created_at}>{new Date(version.created_at).toLocaleString()}</time>
            {version.error && <span className="error-text">{version.error}</span>}
          </div>}
        </div>
      </div>
    </div>
  </article>;
}
