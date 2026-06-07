"use client";

import { useState } from "react";

interface AuthFormProps {
  onLogin: (personId: string | null, password: string, name?: string, identity?: string, email?: string, code?: string) => Promise<void>;
  loading: boolean;
}

export default function AuthForm({ onLogin, loading }: AuthFormProps) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [step, setStep] = useState<0 | 1 | 2>(0);

  // 登录
  const [loginEmail, setLoginEmail] = useState("");
  const [loginPassword, setLoginPassword] = useState("");

  // 注册
  const [regEmail, setRegEmail] = useState("");
  const [regName, setRegName] = useState("");
  const [regIdentity, setRegIdentity] = useState<"面试官" | "普通用户">("面试官");
  const [regCompany, setRegCompany] = useState("");
  const [regJobRole, setRegJobRole] = useState("");
  const [regTech, setRegTech] = useState<string[]>([]);
  const [regWantToKnow, setRegWantToKnow] = useState<string[]>([]);
  const [regPassword, setRegPassword] = useState("");
  const [regCode, setRegCode] = useState("");
  const [codeSent, setCodeSent] = useState(false);
  const [sendingCode, setSendingCode] = useState(false);

  const handleLogin = () => {
    if (!loginPassword || !loginEmail) return;
    onLogin(null, loginPassword, undefined, undefined, loginEmail);
  };

  const handleSendCode = async () => {
    if (!regEmail) return;
    setSendingCode(true);
    try {
      const res = await fetch("/api/auth/send-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: regEmail }),
      });
      if (res.ok) {
        setCodeSent(true);
      } else {
        const data = await res.json().catch(() => ({}));
        alert(data.error || "验证码发送失败");
      }
    } catch {
      alert("网络错误，请稍后重试");
    } finally {
      setSendingCode(false);
    }
  };

  const handleRegister = () => {
    if (!regEmail || !regName || !regPassword || !regCode) return;
    onLogin(null, regPassword, regName, regIdentity, regEmail, regCode);
  };

  const toggleTech = (t: string) => {
    setRegTech((prev) => prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]);
  };
  const toggleWant = (w: string) => {
    setRegWantToKnow((prev) => prev.includes(w) ? prev.filter((x) => x !== w) : [...prev, w]);
  };

  // ====== 登录视图 ======
  if (mode === "login") {
    return (
      <div className="space-y-4">
        <p className="text-sm text-ink-muted">用邮箱登录。</p>

        <input
          type="email"
          placeholder="你的邮箱"
          value={loginEmail}
          onChange={(e) => setLoginEmail(e.target.value)}
          className="w-full bg-cream-hover border-2 border-border px-3 py-2 text-sm"
        />

        <input
          type="password"
          placeholder="输入密码"
          value={loginPassword}
          onChange={(e) => setLoginPassword(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleLogin()}
          className="w-full bg-cream-hover border-2 border-border px-3 py-2 text-sm"
        />

        <button
          onClick={handleLogin}
          disabled={loading || !loginPassword}
          className="w-full pixel-btn py-2 text-sm font-bold tracking-wider"
        >
          {loading ? "◆ 验证中..." : "◆ 登录"}
        </button>

        <p className="text-center text-xs text-ink-muted">
          没有账号？{" "}
          <button
            onClick={() => { setMode("register"); setStep(0); }}
            className="underline hover:text-accent"
          >
            注册
          </button>
        </p>
      </div>
    );
  }

  // ====== 注册 Step 1: 身份选择 ======
  if (step === 0) {
    return (
      <div className="space-y-4">
        <p className="text-sm font-bold tracking-wider">你好～我是 NaNaGi ✨ 请问你是？</p>

        <button
          onClick={() => { setRegIdentity("面试官"); setStep(1); }}
          className="w-full pixel-btn py-3 text-sm"
        >
          🎯 我是来面试南志锦的
        </button>

        <button
          onClick={() => { setRegIdentity("普通用户"); setStep(2); }}
          className="w-full pixel-btn py-3 text-sm"
        >
          👤 我只是随便看看
        </button>

        <p className="text-center text-xs text-ink-muted">
          已有账号？{" "}
          <button
            onClick={() => setMode("login")}
            className="underline hover:text-accent"
          >
            登录
          </button>
        </p>
      </div>
    );
  }

  // ====== 注册 Step 2: 面试官表单 ======
  if (step === 1 && regIdentity === "面试官") {
    const allTechs = ["推荐系统", "GNN/图神经网络", "Agent/AI Agent", "计算机视觉", "NLP", "音频/音乐检索", "后端工程", "数据处理/Spark", "模型部署/MLOps"];
    const allWants = ["GNN社交图谱项目", "CnnMusic多模态召回", "FruitCNN水果识别", "南志锦的整体技术能力", "项目背后的设计决策"];

    return (
      <div className="space-y-3 max-h-[50vh] overflow-y-auto">
        <p className="text-sm font-bold tracking-wider">认识一下你～</p>

        <input
          type="email"
          placeholder="📧 邮箱 (用于登录)"
          value={regEmail}
          onChange={(e) => setRegEmail(e.target.value)}
          className="w-full bg-cream-hover border-2 border-border px-3 py-2 text-sm"
        />

        <input
          type="text"
          placeholder="你的名字"
          value={regName}
          onChange={(e) => setRegName(e.target.value)}
          className="w-full bg-cream-hover border-2 border-border px-3 py-2 text-sm"
        />

        <input
          type="text"
          placeholder="🏢 公司 (例: 字节跳动)"
          value={regCompany}
          onChange={(e) => setRegCompany(e.target.value)}
          className="w-full bg-cream-hover border-2 border-border px-3 py-2 text-sm"
        />

        <input
          type="text"
          placeholder="💼 招聘岗位 (例: Agent应用开发工程师)"
          value={regJobRole}
          onChange={(e) => setRegJobRole(e.target.value)}
          className="w-full bg-cream-hover border-2 border-border px-3 py-2 text-sm"
        />

        <div>
          <p className="text-xs font-bold mb-1">🔍 我关注的技术方向 (可多选)</p>
          <div className="flex flex-wrap gap-1">
            {allTechs.map((t) => (
              <button
                key={t}
                onClick={() => toggleTech(t)}
                className={`px-2 py-0.5 text-xs border-2 ${regTech.includes(t) ? "bg-accent text-white border-accent" : "bg-cream-hover border-border"}`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        <div>
          <p className="text-xs font-bold mb-1">📋 我想了解 (可多选)</p>
          <div className="flex flex-wrap gap-1">
            {allWants.map((w) => (
              <button
                key={w}
                onClick={() => toggleWant(w)}
                className={`px-2 py-0.5 text-xs border-2 ${regWantToKnow.includes(w) ? "bg-accent text-white border-accent" : "bg-cream-hover border-border"}`}
              >
                {w}
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={() => setStep(2)}
          disabled={!regName}
          className="w-full pixel-btn py-2 text-sm font-bold tracking-wider"
        >
          ◆ 下一步
        </button>

        <button onClick={() => setStep(0)} className="w-full text-xs text-ink-muted underline">
          ← 返回
        </button>
      </div>
    );
  }

  // ====== 注册 Step 2 (普通用户) / Step 3 (面试官): 设置密码 ======
  return (
    <div className="space-y-4">
      <p className="text-sm font-bold tracking-wider">
        {regIdentity === "面试官" ? "最后一步～" : "设置你的专属密码"}
      </p>

      {regIdentity === "普通用户" && (
        <>
          <input
            type="email"
            placeholder="📧 邮箱 (用于登录)"
            value={regEmail}
            onChange={(e) => setRegEmail(e.target.value)}
            className="w-full bg-cream-hover border-2 border-border px-3 py-2 text-sm"
          />
          <input
            type="text"
            placeholder="你的名字"
            value={regName}
            onChange={(e) => setRegName(e.target.value)}
            className="w-full bg-cream-hover border-2 border-border px-3 py-2 text-sm"
          />
        </>
      )}

      {/* 验证码 */}
      <div className="flex gap-2">
        <input
          type="text"
          placeholder="📨 邮箱验证码"
          value={regCode}
          onChange={(e) => setRegCode(e.target.value)}
          className="flex-1 bg-cream-hover border-2 border-border px-3 py-2 text-sm font-mono tracking-widest"
          maxLength={6}
        />
        <button
          onClick={handleSendCode}
          disabled={sendingCode || !regEmail || codeSent}
          className="px-3 py-2 text-xs font-bold tracking-wider border-2 border-border bg-cream-hover disabled:opacity-50 whitespace-nowrap"
        >
          {sendingCode ? "发送中..." : codeSent ? "已发送 ✓" : "获取验证码"}
        </button>
      </div>

      <input
        type="password"
        placeholder="🔑 设置密码 (下次用这个登录)"
        value={regPassword}
        onChange={(e) => setRegPassword(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && handleRegister()}
        className="w-full bg-cream-hover border-2 border-border px-3 py-2 text-sm"
      />

      <button
        onClick={handleRegister}
        disabled={loading || !regName || !regPassword}
        className="w-full pixel-btn py-2 text-sm font-bold tracking-wider"
      >
        {loading ? "◆ 注册中..." : "◆ 开始对话 ✨"}
      </button>

      <button
        onClick={() => {
          regIdentity === "面试官" ? setStep(1) : setStep(0);
        }}
        className="w-full text-xs text-ink-muted underline"
      >
        ← 返回
      </button>
    </div>
  );
}
