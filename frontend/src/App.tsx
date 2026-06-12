import { useState, useEffect, useRef } from 'react'
import { Send, Plus, MessageSquare, Trash2, User, Sparkles } from 'lucide-react'
import axios from 'axios'
import Plot from 'react-plotly.js'
interface Session {
 session_id: string;
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
 animate?: boolean;
}

const parseMessageContent = (backendData: any): string => {
  if (!backendData) return 'Analysis complete';
  if (backendData.response) return backendData.response;
  
  const business = backendData.business_response || backendData.business || {};
  if (typeof business === 'string') return business;
  if (business.analysis) return business.analysis;
  if (business.business_insights) return business.business_insights;
  
  const executor = backendData.executor_response || backendData.executor || {};
  if (executor.console_output) return executor.console_output;
  
  return 'Analysis complete';
};

const parseMetadata = (backendData: any) => {
  if (!backendData) return undefined;
  
  const metadata: any = {};
  
  const pythonCode = backendData.coder_response || backendData.python;
  if (pythonCode && typeof pythonCode === 'string' && pythonCode.trim() !== '') {
    metadata.generated_python = pythonCode;
  }
  
  const executor = backendData.executor_response || backendData.executor || {};
  if (executor.plotly_charts && Array.isArray(executor.plotly_charts)) {
    const charts = [];
    for (const c of executor.plotly_charts) {
      let fig = c.figure || c.data;
      if (typeof fig === 'string') {
        try { fig = JSON.parse(fig); } catch(e) {}
      }
      if (fig && fig.data) {
        charts.push(fig);
      }
    }
    if (charts.length > 0) {
      metadata.charts = charts;
    }
  }
  
  return Object.keys(metadata).length > 0 ? metadata : undefined;
};

const TypewriterMessage = ({ content }: { content: string }) => {
 const [displayedContent, setDisplayedContent] = useState('')
 useEffect(() => {
  let i = 0
  const timer = setInterval(() => {
   setDisplayedContent(content.substring(0, i))
   i += 1
   if (i > content.length) {
    clearInterval(timer)
   }
  }, 10)
  return () => clearInterval(timer)
 }, [content])
 return <div dangerouslySetInnerHTML={{ __html: displayedContent.replace(/\\n/g, '<br/>').replace(/\n/g, '<br/>') }} />
}

export default function App() {
 const [sessions, setSessions] = useState<Session[]>([])
 const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
 const [messages, setMessages] = useState<Message[]>([])
 const [inputValue, setInputValue] = useState('')
 const [isStreaming, setIsStreaming] = useState(false)
 const [streamProgress, setStreamProgress] = useState('')
 const [sidebarWidth, setSidebarWidth] = useState(280)
 const [isResizing, setIsResizing] = useState(false)
 const messagesEndRef = useRef<HTMLDivElement>(null)
 const fetchSessions = async () => {
  try {
   const res = await axios.get('/api/agents/sessions')
   setSessions(res.data.sessions || [])
  } catch (e) {
   console.error("Failed to fetch sessions", e)
  }
 }
 useEffect(() => {
  fetchSessions()
 }, [])

 useEffect(() => {
  const handleMouseMove = (e: MouseEvent) => {
   if (!isResizing) return
   // Constrain the width to reasonable bounds (e.g., min 200px, max 600px)
   const newWidth = Math.min(Math.max(e.clientX, 200), 600)
   setSidebarWidth(newWidth)
  }
  const handleMouseUp = () => {
   setIsResizing(false)
  }
  
  if (isResizing) {
   document.addEventListener('mousemove', handleMouseMove)
   document.addEventListener('mouseup', handleMouseUp)
  }
  return () => {
   document.removeEventListener('mousemove', handleMouseMove)
   document.removeEventListener('mouseup', handleMouseUp)
  }
 }, [isResizing])
 const loadChat = async (sessionId: string) => {
  setCurrentSessionId(sessionId)
  try {
   const res = await axios.get(`/api/agents/sessions/${sessionId}`)
   const loadedMessages: Message[] = []
   if (res.data && res.data.conversations && Array.isArray(res.data.conversations)) {
       res.data.conversations.forEach((conv: any) => {
           if (conv.input) {
               loadedMessages.push({ role: 'user', content: conv.input })
           }
            let aiContent = parseMessageContent(conv.agent_responses)
            if (conv.status === 'error') {
                const rawError = (conv.error || '').toLowerCase()
                if (rawError.includes('db_credentials') || rawError.includes('encryption_key') || rawError.includes('database credentials') || rawError.includes('db') || rawError.includes('postgres') || rawError.includes('mongo')) {
                    aiContent = 'Error: Database service is down. Please configure your database credentials.'
                } else if (rawError.includes('llm') || rawError.includes('api_key') || rawError.includes('quota') || rawError.includes('genai')) {
                    aiContent = 'Error: Language model service is unavailable or misconfigured. Please check API keys.'
                } else if (conv.error) {
                    aiContent = 'Error: An internal service error occurred. Please try again later.'
                }
            }
           const metadata = parseMetadata(conv.agent_responses)
           loadedMessages.push({ 
               role: 'assistant', 
               content: aiContent,
               ...(metadata ? { metadata } : {})
           })
       })
   }
   setMessages(loadedMessages)
  } catch (e) {
   console.error("Failed to load messages", e)
  }
 }
 const deleteSession = async (e: React.MouseEvent, sessionId: string) => {
  e.stopPropagation() 
  try {
   await axios.delete(`/api/agents/sessions/${sessionId}`)
   setSessions(prev => prev.filter(s => s.session_id !== sessionId))
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
   let activeSessionId = currentSessionId
   let isNewSession = false
   if (!activeSessionId) {
     activeSessionId = crypto.randomUUID()
     setCurrentSessionId(activeSessionId)
     isNewSession = true
   }
   const url = `/api/agents/agent_query_stream?input=${encodeURIComponent(query)}&session_id=${activeSessionId}`
   const response = await fetch(url, {
    method: 'GET',
    headers: { 'Accept': 'text/event-stream' }
   });
   
   if (isNewSession) {
     // Optimistically fetch sessions after starting the new request, but give the DB time to save
     setTimeout(() => fetchSessions(), 1500)
   }
   if (!response.body) throw new Error("No response body")
   const reader = response.body.getReader()
   const decoder = new TextDecoder()
   let buffer = ''
   while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const events = buffer.split('\n\n')
    buffer = events.pop() || ''
    
    for (const ev of events) {
     if (!ev.trim()) continue
     
     let eventType = ''
     let eventData = ''
     
     const lines = ev.split('\n')
     for (const line of lines) {
      if (line.startsWith('event: ')) eventType = line.substring(7)
      else if (line.startsWith('data: ')) eventData = line.substring(6)
     }
     
     if (eventType === 'session') {
      // Backend confirms session_id — but we trust the local activeSessionId we already set.
      // Do NOT overwrite the session id here — it causes a stale-closure race condition.
      // Just refresh the sidebar list to show the new chat after a short delay.
      try {
       const data = JSON.parse(eventData)
       if (data.session_id && data.session_id !== activeSessionId) {
        // Backend assigned a different id (e.g. we sent empty string before the fix)
        // Adopt it and update
        setCurrentSessionId(data.session_id)
       }
      } catch (e) {}
     }
     else if (eventType === 'progress') {
      try {
       const data = JSON.parse(eventData)
       setStreamProgress(data.log)
      } catch (e) {}
     }
     else if (eventType === 'final') {
      try {
       const data = JSON.parse(eventData)
       const aiContent = parseMessageContent(data)
       const metadata = parseMetadata(data)
       
       setMessages(prev => [...prev, {
        role: 'assistant',
        content: aiContent,
        animate: true,
        ...(metadata ? { metadata } : {})
       }])
       // Refresh sidebar after every successful response so it stays up to date
       setTimeout(() => fetchSessions(), 800)
      } catch (e) {}
     }
     else if (eventType === 'error') {
      try {
       const data = JSON.parse(eventData)
       setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${data.detail || 'Something went wrong.'}`
       }])
       // Also refresh sidebar on error so partial saves appear
       setTimeout(() => fetchSessions(), 800)
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
  <div className="app-container" style={{ 
   cursor: isResizing ? 'col-resize' : 'default',
   userSelect: isResizing ? 'none' : 'auto'
  }}>
   {}
   <div className="sidebar" style={{ width: sidebarWidth }}>
    <div className="sidebar-header">
     <img src="/logo.jpg.webp" alt="Elytics" className="logo-img" />
    </div>
    <button 
     className="new-chat-btn"
     onClick={() => {
      setCurrentSessionId(null)
      setMessages([])
      setIsStreaming(false)
      setStreamProgress('')
      fetchSessions()
     }}
    >
     <Plus size={18} /> New Chat
    </button>
    <div className="history-list">
     {sessions.map(s => (
      <div 
       key={s.session_id} 
       className={`history-item ${s.session_id === currentSessionId ? 'active' : ''}`}
       onClick={() => loadChat(s.session_id)}
      >
       <div style={{ display: 'flex', alignItems: 'center', flex: 1, minWidth: 0 }}>
        <MessageSquare size={14} style={{ flexShrink: 0, marginRight: '6px' }}/>
        <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', flex: 1 }}>
         {s.title}
        </span>
       </div>
       <button 
        className="delete-session-btn"
        onClick={(e) => deleteSession(e, s.session_id)}
        title="Delete Chat"
       >
        <Trash2 size={14} />
       </button>
      </div>
     ))}
    </div>
   </div>
   <div 
    className="sidebar-resizer"
    onMouseDown={(e) => {
     e.preventDefault()
     setIsResizing(true)
    }}
   />
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
         <div className="message-content">
           {msg.role === 'assistant' ? (
            msg.animate ? (
             <TypewriterMessage content={msg.content} />
            ) : (
             <div dangerouslySetInnerHTML={{ __html: msg.content.replace(/\\n/g, '<br/>').replace(/\n/g, '<br/>') }} />
            )
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
