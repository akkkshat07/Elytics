import { useState, useEffect, useRef } from 'react'
import { Send, Menu, Plus, MessageSquare, Code, Terminal, Trash2 } from 'lucide-react'
import axios from 'axios'
import Plot from 'react-plotly.js'
interface Session {
  id: string;
  title: string;
  created_at: string;
}
interface Message {
  id?: number;
  role: 'user' | 'assistant';
  content: string;
  metadata?: {
    generated_sql?: string;
    generated_python?: string;
    charts?: any[];
  };
}
export default function App() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamProgress, setStreamProgress] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fetchSessions = async () => {
    try {
      const res = await axios.get('/api/sessions')
      setSessions(res.data)
    } catch (e) {
      console.error("Failed to fetch sessions", e)
    }
  }
  useEffect(() => {
    fetchSessions()
  }, [])
  const loadChat = async (sessionId: string) => {
    setCurrentSessionId(sessionId)
    try {
      const res = await axios.get(`/api/sessions/${sessionId}/messages`)
      setMessages(res.data)
    } catch (e) {
      console.error("Failed to load messages", e)
    }
  }
  const deleteSession = async (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation() 
    try {
      await axios.delete(`/api/sessions/${sessionId}`)
      setSessions(prev => prev.filter(s => s.id !== sessionId))
      if (currentSessionId === sessionId) {
        setCurrentSessionId(null)
        setMessages([])
      }
    } catch (e) {
      console.error("Failed to delete session", e)
    }
  }
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamProgress])
  const handleSend = async () => {
    if (!inputValue.trim() || isStreaming) return
    const query = inputValue.trim()
    setInputValue('')
    const newMessages = [...messages, { role: 'user' as const, content: query }]
    setMessages(newMessages)
    setIsStreaming(true)
    setStreamProgress('Initializing pipeline...')
    try {
      const response = await fetch('/api/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, session_id: currentSessionId })
      });
      if (!response.body) throw new Error("No response body")
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        const events = chunk.split('\n\n')
        for (const ev of events) {
          if (!ev.trim()) continue
          if (ev.startsWith('event: session')) {
            const dataStr = ev.replace('event: session\ndata: ', '')
            try {
              const data = JSON.parse(dataStr)
              if (!currentSessionId) {
                setCurrentSessionId(data.session_id)
                fetchSessions() 
              }
            } catch (e) {}
          } 
          else if (ev.startsWith('event: progress')) {
            const dataStr = ev.replace('event: progress\ndata: ', '')
            try {
              const data = JSON.parse(dataStr)
              setStreamProgress(data.log)
            } catch (e) {}
          }
          else if (ev.startsWith('event: complete')) {
            const dataStr = ev.replace('event: complete\ndata: ', '')
            try {
              const data = JSON.parse(dataStr)
              setMessages(prev => [...prev, {
                role: 'assistant',
                content: data.insights.join('\n\n'),
                metadata: {
                  generated_sql: data.generated_sql,
                  generated_python: data.generated_python,
                  charts: data.charts
                }
              }])
            } catch (e) {}
          }
          else if (ev.startsWith('event: error')) {
            const dataStr = ev.replace('event: error\ndata: ', '')
            try {
              const data = JSON.parse(dataStr)
              setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Error: ${data.error}`
              }])
            } catch(e) {}
          }
        }
      }
    } catch (e) {
      console.error(e)
      setMessages(prev => [...prev, { role: 'assistant', content: 'Connection failed.' }])
    } finally {
      setIsStreaming(false)
      setStreamProgress('')
    }
  }
  return (
    <div className="app-container">
      {}
      <div className="sidebar">
        <div className="sidebar-header">
          <img src="/logo.jpg.webp" alt="Elytics" className="logo-img" />
        </div>
        <button 
          className="new-chat-btn"
          onClick={() => {
            setCurrentSessionId(null)
            setMessages([])
          }}
        >
          <Plus size={18} /> New Chat
        </button>
        <div className="history-list">
          {sessions.map(s => (
            <div 
              key={s.id} 
              className={`history-item ${s.id === currentSessionId ? 'active' : ''}`}
              onClick={() => loadChat(s.id)}
            >
              <div style={{ display: 'flex', alignItems: 'center', flex: 1, minWidth: 0 }}>
                <MessageSquare size={14} style={{ flexShrink: 0, marginRight: '6px' }}/>
                <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', flex: 1 }}>
                  {s.title}
                </span>
              </div>
              <button 
                className="delete-session-btn"
                onClick={(e) => deleteSession(e, s.id)}
                title="Delete Chat"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      </div>
      {}
      <div className="main-content">
        {messages.length === 0 && !isStreaming ? (
          <div className="welcome-screen">
            <img src="/logo.jpg.webp" alt="Elytics Logo" className="welcome-logo" />
            <h1 className="welcome-title">Welcome to Elytics</h1>
            <p className="welcome-subtitle">Ask me anything about your data.</p>
          </div>
        ) : (
          <div className="chat-scroll-area">
            <div className="chat-container">
              {messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}`}>
                  <div className={`avatar ${msg.role}`}>
                    {msg.role === 'user' ? 'U' : 'E'}
                  </div>
                  <div className="message-content">
                    {msg.role === 'assistant' ? (
                      <div dangerouslySetInnerHTML={{ __html: msg.content.replace(/\\n/g, '<br/>') }} />
                    ) : (
                      msg.content
                    )}
                    {}
                    {msg.metadata?.generated_sql && (
                      <CollapsibleCode title="Generated SQL" code={msg.metadata.generated_sql} icon={<Terminal size={16}/>} />
                    )}
                    {msg.metadata?.generated_python && (
                      <CollapsibleCode title="Analysis Python Code" code={msg.metadata.generated_python} icon={<Code size={16}/>} />
                    )}
                    {}
                    {msg.metadata?.charts && msg.metadata.charts.map((chart, cIdx) => (
                      <div key={cIdx} className="chart-container">
                        <Plot
                          data={chart.data}
                          layout={{ ...chart.layout, autosize: true, margin: { t: 40, r: 20, l: 40, b: 40 } }}
                          useResizeHandler={true}
                          style={{ width: '100%', height: '400px' }}
                          config={{ responsive: true, displayModeBar: false }}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
              {}
              {isStreaming && streamProgress && (
                <div className="message assistant">
                  <div className="avatar ai">E</div>
                  <div className="message-content">
                    <div className="streaming-status">
                      <div className="spinner"></div>
                      {streamProgress}
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>
        )}
        {}
        <div className="input-area">
          <div className="input-container">
            <textarea
              className="chat-input"
              placeholder="Ask a question about your data..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              rows={1}
            />
            <button 
              className="send-btn" 
              onClick={handleSend}
              disabled={!inputValue.trim() || isStreaming}
            >
              <Send size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
function CollapsibleCode({ title, code, icon }: { title: string, code: string, icon: React.ReactNode }) {
  const [isOpen, setIsOpen] = useState(false)
  return (
    <div className="code-collapse">
      <div className="code-collapse-header" onClick={() => setIsOpen(!isOpen)}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          {icon} {title}
        </span>
        <span>{isOpen ? '▲' : '▼'}</span>
      </div>
      {isOpen && (
        <div className="code-collapse-body">
          {code}
        </div>
      )}
    </div>
  )
}
