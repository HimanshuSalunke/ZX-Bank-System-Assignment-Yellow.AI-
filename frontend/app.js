/* ═══════════════════════════════════════════════════════════════════
   ZX Bank AI — Chat Client (SSE Streaming)
   ═══════════════════════════════════════════════════════════════════ */

const API_BASE = '/api';
let sessionId = null;
let isProcessing = false;

// ── DOM Elements ──────────────────────────────────────────────────
const chatContainer = document.getElementById('chatContainer');
const messagesDiv = document.getElementById('messages');
const welcomeCard = document.getElementById('welcomeCard');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const charCount = document.getElementById('charCount');
const statusBadge = document.getElementById('statusBadge');

// ── Initialisation ────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupInputListeners();
});

// ── Health Check ──────────────────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        updateStatus(data.status === 'healthy' ? 'online' : 'degraded', data.status);
    } catch {
        updateStatus('error', 'Offline');
    }
}

function updateStatus(state, text) {
    const dot = statusBadge.querySelector('.status-dot');
    const label = statusBadge.querySelector('.status-text');
    dot.className = `status-dot ${state}`;
    label.textContent = text === 'healthy' ? 'Online' : text.charAt(0).toUpperCase() + text.slice(1);
}

// ── Input Handling ────────────────────────────────────────────────
function setupInputListeners() {
    messageInput.addEventListener('input', () => {
        autoResize();
        const len = messageInput.value.trim().length;
        charCount.textContent = `${messageInput.value.length} / 2000`;
        sendBtn.disabled = len === 0 || isProcessing;
    });

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled) sendMessage();
        }
    });
}

function autoResize() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

// ── Send Message (SSE Streaming) ──────────────────────────────────
async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || isProcessing) return;

    if (welcomeCard) welcomeCard.classList.add('hidden');

    appendMessage('user', text);
    messageInput.value = '';
    messageInput.style.height = 'auto';
    charCount.textContent = '0 / 2000';
    sendBtn.disabled = true;
    isProcessing = true;

    // Create assistant message element for streaming into
    const { el, contentEl } = createAssistantMessage();
    let fullResponse = '';
    let meta = {};

    try {
        const res = await fetch(`${API_BASE}/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, session_id: sessionId }),
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    var currentEvent = line.slice(7).trim();
                } else if (line.startsWith('data: ') && currentEvent) {
                    const data = JSON.parse(line.slice(6));

                    if (currentEvent === 'meta') {
                        meta = data;
                        sessionId = data.session_id;
                    } else if (currentEvent === 'token') {
                        fullResponse += data;
                        contentEl.innerHTML = formatMarkdown(fullResponse);
                        scrollToBottom();
                    } else if (currentEvent === 'done') {
                        // Add sources and meta badges
                        if (meta.sources && meta.sources.length > 0) {
                            let sourcesHtml = '<div class="sources"><div class="sources-label">Sources</div>';
                            meta.sources.forEach(s => {
                                sourcesHtml += `<span class="source-tag">${escapeHtml(s.doc_title)} › ${escapeHtml(s.section)}</span>`;
                            });
                            sourcesHtml += '</div>';
                            contentEl.innerHTML += sourcesHtml;
                        }
                        if (meta.query_type) {
                            let metaHtml = '<div class="message-meta">';
                            metaHtml += `<span class="meta-badge type">${formatType(meta.query_type)}</span>`;
                            if (meta.confidence > 0) {
                                metaHtml += `<span class="meta-badge confidence">${Math.round(meta.confidence * 100)}% confidence</span>`;
                            }
                            metaHtml += '</div>';
                            contentEl.innerHTML += metaHtml;
                        }
                    }
                    currentEvent = null;
                }
            }
        }
    } catch (err) {
        contentEl.innerHTML = formatMarkdown('Sorry, something went wrong. Please try again or call **1800-200-9925**.');
    }

    isProcessing = false;
    sendBtn.disabled = messageInput.value.trim().length === 0;
    messageInput.focus();
}

function sendQuickMessage(text) {
    messageInput.value = text;
    sendMessage();
}

// ── Message Rendering ─────────────────────────────────────────────
function appendMessage(role, content) {
    const el = document.createElement('div');
    el.className = `message ${role}`;
    const avatar = role === 'user' ? '👤' : '🏦';
    el.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">${formatMarkdown(content)}</div>
    `;
    messagesDiv.appendChild(el);
    scrollToBottom();
}

function createAssistantMessage() {
    const el = document.createElement('div');
    el.className = 'message assistant';
    el.innerHTML = `
        <div class="message-avatar">🏦</div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    messagesDiv.appendChild(el);
    scrollToBottom();
    const contentEl = el.querySelector('.message-content');
    return { el, contentEl };
}

// ── Helpers ───────────────────────────────────────────────────────
function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function formatMarkdown(text) {
    if (!text) return '';
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code>$1</code>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/(<li>.+<\/li>\n?)+/g, '<ul>$&</ul>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/^/, '<p>')
        .replace(/$/, '</p>');
}

function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}

function formatType(type) {
    const map = {
        document_query: '📄 Document',
        small_talk: '💬 Chat',
        escalation: '🤝 Escalation',
        adversarial: '🛡️ Safety',
        error: '⚠️ Error',
        unknown: '❓ Unknown',
    };
    return map[type] || type;
}
