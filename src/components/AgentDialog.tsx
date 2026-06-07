"use client";

import { useEffect, useRef } from "react";
import { useChat } from "@/contexts/ChatContext";
import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";
import AuthForm from "./AuthForm";

// Full-screen central dialog for the home page
export default function AgentDialog() {
  const {
    messages,
    streaming,
    isAuthenticated,
    checking,
    loginError,
    loginLoading,
    login,
    sendMessage,
  } = useChat();

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // —— Checking ——
  if (checking) {
    return (
      <div className="rpg-dialog w-full max-w-2xl mx-4">
        <p className="text-ink-muted text-sm animate-pulse">
          ◆ 系统启动中...
        </p>
      </div>
    );
  }

  // —— Locked: password prompt ——
  if (!isAuthenticated) {
    return (
      <div className="w-full max-w-2xl mx-4">
        <div className="rpg-dialog">
          <div className="text-center mb-6">
            <h1 className="text-2xl font-bold tracking-wider mb-2">NaNaGi</h1>
            <p className="text-ink-muted text-sm tracking-wider">
              ◆ 专属女仆 · AI向导 ◆
            </p>
          </div>

          <div className="msg-agent mb-5">
            <div className="text-xs font-bold mb-1 opacity-50 tracking-wider">
              ◆ NaNaGi
            </div>
            <p className="text-sm leading-relaxed">
              欢迎光临～我是NaNaGi，主人的专属女仆 ✨{"\n\n"}登录或注册，让我来招待您。
            </p>
          </div>

          <AuthForm onLogin={login} loading={loginLoading} />

          {loginError && (
            <p className="text-accent text-xs mt-3 font-bold tracking-wider">
              ✗ {loginError}
            </p>
          )}
          {loginLoading && (
            <p className="text-ink-muted text-xs mt-3 animate-pulse tracking-wider">
              ◆ 验证中...
            </p>
          )}
        </div>

        <p className="text-center text-ink-muted text-xs mt-6 tracking-wider">
          仅供面试使用 · 密码请联系南志锦获取
        </p>
      </div>
    );
  }

  // —— Authenticated: full chat ——
  return (
    <div
      className="w-full max-w-2xl mx-4 flex flex-col"
      style={{ height: "calc(100vh - 4rem)" }}
    >
      {/* Header */}
      <div className="pixel-border-light bg-cream-card px-4 py-2 mb-3 flex items-center justify-between">
        <span className="text-xs font-bold tracking-wider">◆ NaNaGi 在线</span>
        <span className="text-xs text-ink-muted">
          {streaming ? "回复中..." : "就绪"}
        </span>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-1 py-2 space-y-3 mb-3">
        {messages.length === 0 && (
          <div className="msg-agent">
            <div className="text-xs font-bold mb-1 opacity-50 tracking-wider">
              ◆ NaNaGi
            </div>
            <p className="text-sm leading-relaxed">
              欢迎光临～我是NaNaGi，主人的专属女仆 ✨
              {"\n\n"}
              让我来介绍一下主人吧！主人是一名AI/ML工程师，技术栈覆盖机器学习全流程：从数据分析、特征工程到模型训练和部署落地。
              {"\n\n"}
              主人有 3 个很厉害的项目：
              {"\n"}
              • 🍎 水果识别CNN — ONNX浏览器端实时推理
              {"\n"}
              • 🎵 CnnMusic — 多模态音频内容召回
              {"\n"}
              • 🔗 GNN社交图谱链接预测 — CAAI-BDSC2023
              {"\n\n"}
              您想了解哪个呢？或者让我直接带您去看看？
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <ChatMessage
            key={i}
            role={msg.role}
            content={msg.content}
            imageUrl={msg.imageUrl}
            isStreaming={
              streaming && i === messages.length - 1 && msg.role === "agent"
            }
          />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="pixel-border-light bg-cream-card px-4 py-3">
        <ChatInput
          onSend={sendMessage}
          disabled={streaming}
          placeholder="问任何关于南志锦或项目的问题..."
        />
        <p className="text-ink-muted text-xs mt-2 text-center tracking-wider">
          Enter 发送 · NaNaGi
        </p>
      </div>
    </div>
  );
}
