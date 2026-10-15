import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Check, Copy } from 'lucide-react';

function citations(count) {
  return () => tree => {
    const walk = node => {
      if (!node.children || ['link', 'code', 'inlineCode'].includes(node.type)) return;
      node.children = node.children.flatMap(child => {
        if (child.type !== 'text') { walk(child); return [child]; }
        const pieces = [];
        let start = 0;
        for (const match of child.value.matchAll(/\[(\d+)\]/g)) {
          const number = Number(match[1]);
          if (number < 1 || number > count) continue;
          if (match.index > start) pieces.push({ type: 'text', value: child.value.slice(start, match.index) });
          pieces.push({ type: 'link', url: `#source-${number}`, children: [{ type: 'text', value: String(number) }] });
          start = match.index + match[0].length;
        }
        if (!pieces.length) return [child];
        if (start < child.value.length) pieces.push({ type: 'text', value: child.value.slice(start) });
        return pieces;
      });
    };
    walk(tree);
  };
}

function CodeBlock({ children }) {
  const [copied, setCopied] = useState(false);
  const language = children?.props?.className?.replace('language-', '') || 'text';
  async function copy() {
    try {
      await navigator.clipboard.writeText(String(children?.props?.children || ''));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch { setCopied(false); }
  }
  return <div className="code-block"><div className="code-heading"><span>{language}</span>
    <button onClick={copy} aria-label="Copy code">{copied ? <Check size={14} /> : <Copy size={14} />}{copied ? 'Copied' : 'Copy code'}</button>
  </div><pre>{children}</pre></div>;
}

export default function Markdown({ content, sources = [], onSource }) {
  return <div className="markdown"><ReactMarkdown skipHtml remarkPlugins={[remarkGfm, citations(sources.length)]}
    components={{
      pre: CodeBlock,
      a: ({ href, children }) => href?.startsWith('#source-')
        ? <button className="citation" aria-label={`View source ${href.split('-').at(-1)}`} onClick={() => onSource(Number(href.split('-').at(-1)))}>{children}</button>
        : <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>,
      img: ({ alt }) => <span className="omitted-image">{alt || 'Document image'}</span>,
    }}>{content}</ReactMarkdown></div>;
}
