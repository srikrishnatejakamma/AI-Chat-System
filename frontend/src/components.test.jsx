import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createRef } from 'react';
import Response from './Response.jsx';
import Composer from './Composer.jsx';
import Markdown from './Markdown.jsx';
import Sidebar from './Sidebar.jsx';

function version(content, status = 'complete') {
  return { id: content, content, status, model: 'phi3', created_at: '2026-10-15T12:00:00Z', sources: [{ number: 1, name: 'Policy.txt', text: 'Budget: 30,000.' }] };
}

describe('Response controls', () => {
  it('navigates versions and disables arrows at boundaries', async () => {
    const user = userEvent.setup();
    const onVersion = vi.fn();
    const message = { id: 'answer', versions: [version('Original'), version('Revised')], selected_version: 1 };
    const props = { message, busy: false, onVersion, onRegenerate: vi.fn(), onSources: vi.fn(), notify: vi.fn() };
    const view = render(<Response {...props} />);
    expect(screen.getByLabelText('Response version 2 of 2')).toHaveTextContent('2/2');
    expect(screen.getByRole('button', { name: 'Next response' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Previous response' }));
    expect(onVersion).toHaveBeenCalledWith('answer', 0);
    view.rerender(<Response {...props} message={{ ...message, selected_version: 0 }} />);
    expect(screen.getByRole('button', { name: 'Previous response' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Next response' })).toBeEnabled();
    view.rerender(<Response {...props} busy />);
    expect(screen.getByRole('button', { name: 'Previous response' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Regenerate response' })).toBeDisabled();
  });

  it('retains partial answers and retries stopped responses', async () => {
    const retry = vi.fn();
    render(<Response message={{ id: 'answer', versions: [version('Partial text', 'stopped')], selected_version: 0 }} onRegenerate={retry} onSources={vi.fn()} notify={vi.fn()} />);
    expect(screen.getByText('Partial text')).toBeInTheDocument();
    expect(screen.getByText('Response stopped.')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Retry response' }));
    expect(retry).toHaveBeenCalledWith('answer');
  });

  it('copies a response and opens its evidence', async () => {
    const user = userEvent.setup();
    const clipboard = vi.spyOn(navigator.clipboard, 'writeText').mockResolvedValue();
    const onSources = vi.fn();
    const current = version('The budget is 30,000 [1].');
    render(<Response message={{ id: 'answer', versions: [current], selected_version: 0 }} onSources={onSources} onRegenerate={vi.fn()} notify={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: 'Copy response' }));
    expect(clipboard).toHaveBeenCalledWith(current.content);
    await user.click(screen.getByRole('button', { name: 'View source 1' }));
    expect(onSources).toHaveBeenCalledWith(current.sources, 1);
  });
});

describe('Message composer', () => {
  const props = () => ({ draft: 'Budget?', setDraft: vi.fn(), onSend: vi.fn(), onStop: vi.fn(), generating: false,
    disabled: false, uploading: false, documents: [{ id: 'policy', name: 'Policy.txt' }], scope: 'all', setScope: vi.fn(),
    onUpload: vi.fn(), onLibrary: vi.fn(), attachments: [], onRemoveAttachment: vi.fn(), inputRef: createRef() });

  it('sends with Enter and keeps Shift+Enter for a newline', () => {
    const options = props();
    render(<Composer {...options} />);
    const input = screen.getByRole('textbox', { name: 'Message Local Chat' });
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: true });
    expect(options.onSend).not.toHaveBeenCalled();
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(options.onSend).toHaveBeenCalledTimes(1);
  });

  it('offers Stop instead of Send during generation', async () => {
    const options = props();
    render(<Composer {...options} generating />);
    expect(screen.queryByRole('button', { name: 'Send message' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Stop response' }));
    expect(options.onStop).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });
    expect(options.onSend).not.toHaveBeenCalled();
  });

  it('opens the real upload and library actions', async () => {
    const options = props();
    render(<Composer {...options} />);
    await userEvent.click(screen.getByRole('button', { name: 'Add documents' }));
    await userEvent.click(screen.getByRole('button', { name: 'Upload a document' }));
    expect(options.onUpload).toHaveBeenCalledTimes(1);
  });
});

it('renders document markdown without fetching remote images or executing HTML', () => {
  render(<Markdown content={'<script>alert(1)</script>\n\n![Remote image](https://example.com/image.png)\n\nEvidence [1].'} sources={[{}]} onSource={vi.fn()} />);
  expect(document.querySelector('script')).toBeNull();
  expect(document.querySelector('img')).toBeNull();
  expect(screen.getByRole('button', { name: 'View source 1' })).toBeInTheDocument();
});

it('filters saved chats through the sidebar search', async () => {
  render(<Sidebar open chats={[{ id: 'one', title: 'Learning budget' }, { id: 'two', title: 'Travel policy' }]} onToggle={vi.fn()} onNew={vi.fn()} onSelect={vi.fn()} onLibrary={vi.fn()} onSettings={vi.fn()} onRename={vi.fn()} onDelete={vi.fn()} />);
  await userEvent.click(screen.getByRole('button', { name: 'Search chats' }));
  await userEvent.type(screen.getByRole('textbox', { name: 'Search conversations' }), 'learning');
  expect(screen.getByRole('button', { name: 'Learning budget' })).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Travel policy' })).not.toBeInTheDocument();
});
