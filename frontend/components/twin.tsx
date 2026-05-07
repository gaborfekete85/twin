'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Plus, Sparkles, MessageSquare, ChevronRight } from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const SUGGESTIONS = [
  'Tell me about yourself',
  'What can you help me with?',
  'Explain AI deployment',
  'What is a digital twin?',
];

export default function Twin() {
  const [messages, setMessages]     = useState<Message[]>([]);
  const [input, setInput]           = useState('');
  const [isLoading, setIsLoading]   = useState(false);
  const [sessionId, setSessionId]   = useState<string>('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef              = useRef<HTMLDivElement>(null);
  const textareaRef                 = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 160) + 'px';
  }, [input]);

  const sendMessage = async (text?: string) => {
    const content = (text ?? input).trim();
    if (!content || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/chat`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: content, session_id: sessionId || undefined }),
        }
      );

      if (!response.ok) throw new Error('Failed to send message');

      const data = await response.json();
      if (!sessionId) setSessionId(data.session_id);

      setMessages(prev => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: data.response,
          timestamp: new Date(),
        },
      ]);
    } catch {
      setMessages(prev => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const startNewChat = () => {
    setMessages([]);
    setSessionId('');
    setInput('');
  };

  return (
    <div className="flex h-full w-full overflow-hidden">

      {/* ── Sidebar ── */}
      <aside
        className={`flex flex-col shrink-0 bg-[#0d0d1f] border-r border-[#1e1e38] transition-all duration-300 ${
          sidebarOpen ? 'w-64' : 'w-0 overflow-hidden'
        }`}
      >
        {/* Brand */}
        <div className="flex items-center gap-3 px-4 py-5 border-b border-[#1e1e38]">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-violet-900/40">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div>
            <p className="text-sm font-semibold text-white leading-tight">Digital Twin</p>
            <p className="text-[10px] text-slate-500">AI in Production</p>
          </div>
        </div>

        {/* New Chat */}
        <div className="px-3 pt-4">
          <button
            onClick={startNewChat}
            className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg text-sm text-slate-300 hover:bg-[#1a1a30] hover:text-white transition-colors border border-[#2a2a4a] hover:border-violet-600/50"
          >
            <Plus className="w-4 h-4" />
            New chat
          </button>
        </div>

        {/* Recent chats placeholder */}
        <div className="flex-1 px-3 pt-4 overflow-y-auto">
          <p className="text-[10px] uppercase tracking-widest text-slate-600 px-2 mb-2">Recent</p>
          {messages.length > 0 ? (
            <button className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-slate-400 hover:bg-[#1a1a30] hover:text-white transition-colors text-left truncate">
              <MessageSquare className="w-3.5 h-3.5 shrink-0 text-violet-500" />
              <span className="truncate">
                {messages.find(m => m.role === 'user')?.content ?? 'Current chat'}
              </span>
            </button>
          ) : (
            <p className="text-xs text-slate-600 px-2">No recent chats</p>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-4 border-t border-[#1e1e38]">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-violet-500 to-indigo-500 flex items-center justify-center">
              <User className="w-4 h-4 text-white" />
            </div>
            <div>
              <p className="text-xs font-medium text-white">You</p>
              <p className="text-[10px] text-slate-500">Week 2 — Twin</p>
            </div>
          </div>
        </div>
      </aside>

      {/* ── Main area ── */}
      <div className="flex flex-col flex-1 min-w-0 bg-[#0a0a14]">

        {/* Top bar */}
        <header className="flex items-center gap-3 px-4 py-3 border-b border-[#1e1e38] bg-[#0a0a14]/80 backdrop-blur-sm">
          <button
            onClick={() => setSidebarOpen(o => !o)}
            className="p-1.5 rounded-md text-slate-500 hover:text-white hover:bg-[#1a1a30] transition-colors"
            aria-label="Toggle sidebar"
          >
            <ChevronRight className={`w-4 h-4 transition-transform duration-300 ${sidebarOpen ? 'rotate-180' : ''}`} />
          </button>

          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-md bg-gradient-to-br from-violet-600 to-indigo-600 flex items-center justify-center">
              <Bot className="w-3.5 h-3.5 text-white" />
            </div>
            <span className="text-sm font-medium text-white">AI Digital Twin</span>
          </div>

          <div className="ml-auto flex items-center gap-1.5">
            <span className="inline-flex items-center gap-1 text-[11px] text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              Online
            </span>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">

            {/* Empty state */}
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center min-h-[60vh] text-center space-y-6">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-600 to-indigo-600 flex items-center justify-center shadow-2xl shadow-violet-900/50">
                  <Sparkles className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white mb-2">How can I help you?</h2>
                  <p className="text-slate-400 text-sm">Ask me anything — I&apos;m your AI Digital Twin.</p>
                </div>

                {/* Suggestion chips */}
                <div className="grid grid-cols-2 gap-2 w-full max-w-md">
                  {SUGGESTIONS.map(s => (
                    <button
                      key={s}
                      onClick={() => sendMessage(s)}
                      className="px-4 py-3 rounded-xl bg-[#13132a] border border-[#2a2a4a] text-sm text-slate-300 hover:border-violet-600/60 hover:text-white hover:bg-[#1a1a35] transition-all text-left"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Message list */}
            {messages.map(message => (
              <div
                key={message.id}
                className={`flex gap-3 message-enter ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                {/* Assistant avatar */}
                {message.role === 'assistant' && (
                  <div className="shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-violet-900/30 mt-0.5">
                    <Bot className="w-4.5 h-4.5 text-white" />
                  </div>
                )}

                <div
                  className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                    message.role === 'user'
                      ? 'bg-gradient-to-br from-violet-600 to-indigo-600 text-white rounded-tr-sm shadow-lg shadow-violet-900/30'
                      : 'bg-[#13132a] border border-[#2a2a4a] text-slate-200 rounded-tl-sm'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>
                  <p className={`text-[10px] mt-1.5 ${message.role === 'user' ? 'text-violet-200/70' : 'text-slate-600'}`}>
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </p>
                </div>

                {/* User avatar */}
                {message.role === 'user' && (
                  <div className="shrink-0 w-8 h-8 rounded-lg bg-[#1e1e38] border border-[#2a2a4a] flex items-center justify-center mt-0.5">
                    <User className="w-4 h-4 text-slate-300" />
                  </div>
                )}
              </div>
            ))}

            {/* Typing indicator */}
            {isLoading && (
              <div className="flex gap-3 justify-start message-enter">
                <div className="shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-violet-900/30">
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <div className="bg-[#13132a] border border-[#2a2a4a] rounded-2xl rounded-tl-sm px-4 py-3.5">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 bg-violet-500 rounded-full animate-bounce" />
                    <span className="w-2 h-2 bg-violet-400 rounded-full animate-bounce delay-100" />
                    <span className="w-2 h-2 bg-violet-300 rounded-full animate-bounce delay-200" />
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input area */}
        <div className="border-t border-[#1e1e38] bg-[#0a0a14] px-4 py-4">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-end gap-3 bg-[#13132a] border border-[#2a2a4a] rounded-2xl px-4 py-3 focus-within:border-violet-600/60 transition-colors">
              <textarea
                ref={textareaRef}
                rows={1}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Message your Digital Twin..."
                disabled={isLoading}
                className="flex-1 bg-transparent text-sm text-slate-200 placeholder-slate-600 resize-none outline-none leading-relaxed max-h-40 scrollbar-none"
              />
              <button
                onClick={() => sendMessage()}
                disabled={!input.trim() || isLoading}
                className="shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-indigo-600 flex items-center justify-center hover:from-violet-500 hover:to-indigo-500 disabled:opacity-30 disabled:cursor-not-allowed transition-all shadow-lg shadow-violet-900/30"
              >
                <Send className="w-4 h-4 text-white" />
              </button>
            </div>
            <p className="text-center text-[11px] text-slate-700 mt-2">
              Press <kbd className="px-1 py-0.5 rounded bg-[#1e1e38] text-slate-500 text-[10px]">Enter</kbd> to send &nbsp;·&nbsp;
              <kbd className="px-1 py-0.5 rounded bg-[#1e1e38] text-slate-500 text-[10px]">Shift+Enter</kbd> for new line
            </p>
          </div>
        </div>

      </div>
    </div>
  );
}
