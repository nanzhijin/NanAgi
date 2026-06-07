"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useRef,
  useEffect,
  type ReactNode,
} from "react";
import { useRouter } from "next/navigation";
import type {
  Message,
  SSEEvent,
  JukeboxState,
  ModelResultData,
} from "@/lib/types";

const STORAGE_KEY = "nanagi_chat";

// —— Types ——

interface ChatContextType {
  // Auth
  isAuthenticated: boolean;
  userRole: "guest" | "admin" | null;
  checking: boolean;
  loginError: string;
  loginLoading: boolean;
  login: (personId: string | null, password: string, name?: string, identity?: string, email?: string, code?: string) => Promise<void>;

  // Chat
  messages: Message[];
  streaming: boolean;
  sendMessage: (text: string) => Promise<void>;

  // V2: Jukebox + Model Result
  jukebox: JukeboxState;
  modelResult: ModelResultData | null;
  clearModelResult: () => void;

  // Project context
  projectSlug: string | null;
  setProjectSlug: (slug: string | null) => void;

  // Memory panel auto-refresh
  memoryVersion: number;
}

const WELCOME_MESSAGE: Message = {
  id: "welcome",
  role: "agent",
  content:
    "欢迎光临～我是NaNaGi，主人的专属女仆 ✨\n\n请输入面试密码，让我来招待您。",
};

let msgCounter = 0;
function nextId(): string {
  return `msg_${++msgCounter}_${Date.now()}`;
}

// —— Context ——

const ChatContext = createContext<ChatContextType | null>(null);

export function ChatProvider({ children }: { children: ReactNode }) {
  const router = useRouter();

  // Auth
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userRole, setUserRole] = useState<"guest" | "admin" | null>(null);
  const [checking, setChecking] = useState(true);
  const [loginError, setLoginError] = useState("");
  const [loginLoading, setLoginLoading] = useState(false);

  // Chat
  const [messages, setMessages] = useState<Message[]>([WELCOME_MESSAGE]);
  const [streaming, setStreaming] = useState(false);
  const messagesRef = useRef<Message[]>([]);
  const streamAbortRef = useRef<AbortController | null>(null);

  // V2: Jukebox + Model Result
  const [jukebox, setJukebox] = useState<JukeboxState>({
    status: "idle",
    mode: null,
    imageUrl: null,
  });
  const [modelResult, setModelResult] = useState<ModelResultData | null>(null);

  // Project context
  const [projectSlug, setProjectSlug] = useState<string | null>(null);

  // Memory panel auto-refresh
  const [memoryVersion, setMemoryVersion] = useState(0);

  // Keep ref in sync
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  // 聊天记录持久化到 sessionStorage（跨页面导航保持）
  useEffect(() => {
    if (isAuthenticated && messages.length > 0) {
      try {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
      } catch { /* storage full — silently ignore */ }
    }
  }, [messages, isAuthenticated]);

  // Check auth on mount — restore saved messages if already authenticated
  useEffect(() => {
    fetch("/api/auth")
      .then((r) => r.json())
      .then((data) => {
        setIsAuthenticated(data.authenticated);
        setUserRole(data.role || null);
        if (data.authenticated) {
          // 尝试从 sessionStorage 恢复聊天记录
          try {
            const saved = sessionStorage.getItem(STORAGE_KEY);
            if (saved) {
              const parsed = JSON.parse(saved) as Message[];
              if (parsed.length > 0) {
                setMessages(parsed);
                return;
              }
            }
          } catch { /* ignore corrupt storage */ }
          // 没有历史记录 → 清空欢迎消息，开始新对话
          setMessages([]);
        }
      })
      .catch(() => setIsAuthenticated(false))
      .finally(() => setChecking(false));
  }, []);

  // —— Login / Register ——
  const login = useCallback(
    async (personId: string | null, password: string, name?: string, identity?: string, email?: string, code?: string) => {
      setLoginLoading(true);
      setLoginError("");

      // 注册
      if (!personId && name && identity && email) {
        try {
          const res = await fetch("/api/auth/register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, name, identity, password, code }),
          });
          if (res.ok) {
            const data = await res.json();
            setIsAuthenticated(true);
            setUserRole(data.role || "guest");
            setMessages([]);
          } else {
            const data = await res.json().catch(() => ({}));
            setLoginError(data.error || "注册失败");
          }
        } catch {
          setLoginError("连接失败，请稍后重试");
        } finally {
          setLoginLoading(false);
        }
        return;
      }

      // 登录
      try {
        const res = await fetch("/api/auth", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            personId: personId || undefined,
            email: email || undefined,
            password,
          }),
        });

        if (res.ok) {
          const data = await res.json();
          setIsAuthenticated(true);
          setUserRole(data.role || "guest");
          setMessages([]);
        } else {
          const data = await res.json().catch(() => ({}));
          setLoginError(data.error || "密码错误");
      }
    } catch {
      setLoginError("连接失败，请稍后重试");
    } finally {
      setLoginLoading(false);
    }
  }, []);

  // —— Send Message (V2: structured SSE) ——
  const sendMessage = useCallback(
    async (text: string) => {
      if (streaming) return;

      // Abort previous stream if any
      streamAbortRef.current?.abort();
      const abortController = new AbortController();
      streamAbortRef.current = abortController;

      const userMsg: Message = { id: nextId(), role: "user", content: text };
      const agentMsg: Message = { id: nextId(), role: "agent", content: "" };

      setMessages((prev) => {
        const filtered = prev[0]?.id === "welcome" ? [] : prev;
        return [...filtered, userMsg, agentMsg];
      });
      setStreaming(true);
      // 🔮 新一轮对话：清空上一轮的图片和结果
      setJukebox({ status: "idle", mode: null, imageUrl: null });
      setModelResult(null);

      // V2: accumulated agent text content (needs to be outside try for helper access)
      let agentContent = "";

      // —— SSE Event Handlers (defined here for access to agentContent) ——
      function handleSSEEvent(event: SSEEvent) {
        console.log("[ChatContext] SSE事件:", event.type, event.type === "action" ? (event as {action: string}).action : "");
        switch (event.type) {
          case "text":
            agentContent += event.content;
            updateAgentContent(agentContent);
            break;

          case "action":
            if (event.action === "jukebox") {
              const payload = event.payload as {
                status: JukeboxState["status"];
                mode?: JukeboxState["mode"];
                imageUrl?: string;
              };
              setJukebox((prev) => ({
                ...prev,
                ...payload,
              }));
              // 图片生成 → 挂到最新一条 agent 消息上，显示在聊天气泡里
              if (payload.status === "show_image" && payload.imageUrl) {
                console.log("[ChatContext] 📸 show_image 挂载到消息:", payload.imageUrl.slice(0, 80));
                setMessages((prev) => {
                  const copy = [...prev];
                  for (let i = copy.length - 1; i >= 0; i--) {
                    if (copy[i].role === "agent") {
                      copy[i] = { ...copy[i], imageUrl: payload.imageUrl };
                      console.log("[ChatContext] 图片挂到消息 #", i, "role:", copy[i].role);
                      break;
                    }
                  }
                  return copy;
                });
              }
            } else if (event.action === "memory_updated") {
              // 新记忆写入 → 触发面板刷新
              setMemoryVersion((v) => v + 1);
            } else if (event.action === "navigate") {
              const payload = event.payload as { href: string; text: string };
              router.push(payload.href);
            }
            break;

          case "model_result":
            setModelResult(event.result);
            break;
        }
      }

      function updateAgentContent(content: string) {
        setMessages((prev) => {
          const copy = [...prev];
          if (copy.length > 0 && copy[copy.length - 1].role === "agent") {
            copy[copy.length - 1] = {
              ...copy[copy.length - 1],
              content,
            };
          }
          return copy;
        });
      }

      try {
        const history = messagesRef.current.filter((m) => m.id !== "welcome");
        const apiMessages = [...history, userMsg].map((m) => ({
          role: m.role === "agent" ? "assistant" : "user",
          content: m.content,
        }));

        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: apiMessages,
            project: projectSlug, // V2: 传递当前项目上下文
          }),
          signal: abortController.signal,
        });

        if (res.status === 401) {
          setIsAuthenticated(false);
          setMessages([WELCOME_MESSAGE]);
          setStreaming(false);
          return;
        }

        if (!res.ok) {
          const errorText = await res.text();
          setMessages((prev) => {
            const copy = [...prev];
            copy[copy.length - 1] = {
              ...copy[copy.length - 1],
              content: `抱歉，${errorText || "AI服务暂时不可用"}`,
            };
            return copy;
          });
          setStreaming(false);
          return;
        }

        // —— V2: SSE JSON 事件流解析 ——
        const reader = res.body?.getReader();
        if (!reader) {
          setStreaming(false);
          return;
        }

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith("data: ")) continue;

            const jsonStr = trimmed.slice(6);
            if (jsonStr === "[DONE]") continue;

            try {
              const event: SSEEvent = JSON.parse(jsonStr);
              handleSSEEvent(event);
            } catch {
              // Skip unparseable lines (legacy plain-text compatibility)
              // If old format: treat as text
              if (jsonStr && !jsonStr.startsWith("{")) {
                agentContent += jsonStr;
                updateAgentContent(agentContent);
              }
            }
          }
        }
      } catch (err: unknown) {
        if (err instanceof Error && err.name === "AbortError") return;
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = {
            ...copy[copy.length - 1],
            content: "抱歉，连接中断，请重试。",
          };
          return copy;
        });
      } finally {
        setStreaming(false);
        streamAbortRef.current = null;
      }
    },
    [streaming, projectSlug]
  );

  const clearModelResult = useCallback(() => {
    setModelResult(null);
    setJukebox({ status: "idle", mode: null, imageUrl: null });
  }, []);

  return (
    <ChatContext.Provider
      value={{
        isAuthenticated,
        userRole,
        checking,
        loginError,
        loginLoading,
        login,
        messages,
        streaming,
        sendMessage,
        jukebox,
        modelResult,
        clearModelResult,
        projectSlug,
        setProjectSlug,
        memoryVersion,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChat() {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChat must be used within ChatProvider");
  return ctx;
}
