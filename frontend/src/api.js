export async function request(path, options = {}) {
  const { body, ...rest } = options;
  const response = await fetch(`/api${path}`, {
    ...rest,
    ...(body instanceof FormData
      ? { body }
      : body !== undefined ? { body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' } } : {}),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = Array.isArray(data.detail) ? data.detail.map(item => item.msg).join('. ') : data.detail;
    throw new Error(message || `Request failed (${response.status}).`);
  }
  return data;
}

export function downloadChat(chat) {
  return request(`/chats/${chat.id}/export`).then(data => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `${chat.title.replace(/[^a-z0-9 -]/gi, '').slice(0, 60) || 'conversation'}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
}

export function modelName(model) {
  if (model?.startsWith('phi3')) return 'Phi-3';
  if (model?.startsWith('llama3.1')) return 'Llama 3.1';
  return model?.replace(':latest', '') || 'Local model';
}

export function hasModel(status, model) {
  const normalize = value => value.includes(':') ? value : `${value}:latest`;
  return Boolean(status?.online && status.installed_models?.some(value => normalize(value) === normalize(model)));
}
