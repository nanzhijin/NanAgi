// ============================================================
// NaNaGi 邮件服务 — P3 邮箱验证
// nodemailer + SMTP (QQ邮箱默认, 可通过 env 切换)
// ============================================================

import nodemailer from "nodemailer";
import { getEnv } from "./env";

// ==================== 验证码存储 (内存, 5分钟TTL) ====================

interface CodeEntry {
  code: string;
  email: string;
  expiresAt: number;
}

const codeStore = new Map<string, CodeEntry>();
const CODE_TTL_MS = 5 * 60 * 1000; // 5 分钟

/** 生成 6 位数字验证码 */
function generateCode(): string {
  return String(Math.floor(100000 + Math.random() * 900000));
}

/** 存储验证码 */
export function storeCode(email: string): string {
  const code = generateCode();
  const key = email.toLowerCase();
  codeStore.set(key, { code, email, expiresAt: Date.now() + CODE_TTL_MS });
  return code;
}

/** 验证验证码 */
export function verifyCode(email: string, code: string): boolean {
  const key = email.toLowerCase();
  const entry = codeStore.get(key);
  if (!entry) return false;
  if (Date.now() > entry.expiresAt) {
    codeStore.delete(key);
    return false;
  }
  if (entry.code !== code) return false;
  codeStore.delete(key); // 一次性
  return true;
}

// 定期清理过期验证码
setInterval(() => {
  const now = Date.now();
  for (const [key, entry] of codeStore) {
    if (now > entry.expiresAt) codeStore.delete(key);
  }
}, 60_000);

// ==================== SMTP 发送 ====================

let _transporter: nodemailer.Transporter | null = null;

function getTransporter(): nodemailer.Transporter | null {
  if (_transporter) return _transporter;

  const env = getEnv();
  const host = process.env.SMTP_HOST || "smtp.qq.com";
  const port = Number(process.env.SMTP_PORT) || 465;
  const user = process.env.SMTP_USER || "";
  const pass = process.env.SMTP_PASS || "";

  if (!user || !pass) {
    console.warn("[Email] SMTP 未配置 — 验证码功能不可用");
    return null;
  }

  _transporter = nodemailer.createTransport({
    host,
    port,
    secure: port === 465, // 465=SSL, 587=STARTTLS
    requireTLS: port === 587,
    auth: { user, pass },
  });

  console.log(`[Email] SMTP ready: ${user}@${host}`);
  return _transporter;
}

/** 发送验证码邮件 */
export async function sendVerificationCode(email: string): Promise<{ success: boolean; error?: string }> {
  const transporter = getTransporter();
  if (!transporter) {
    return { success: false, error: "邮件服务未配置" };
  }

  const code = storeCode(email);

  try {
    await transporter.sendMail({
      from: process.env.SMTP_FROM || process.env.SMTP_USER || "",
      to: email,
      subject: "NaNaGi 邮箱验证码",
      text: `你的 NaNaGi 验证码是: ${code}\n\n5 分钟内有效。\n\n— NaNaGi 🦊`,
      html: `
        <div style="font-family:monospace;max-width:400px;margin:0 auto;padding:20px;border:2px solid #d4a574;background:#faf3e8;">
          <h2 style="color:#5c3d2e;">NaNaGi 邮箱验证 🦊</h2>
          <p>你的验证码是:</p>
          <p style="font-size:32px;font-weight:bold;letter-spacing:8px;text-align:center;color:#8b4513;">${code}</p>
          <p style="color:#888;font-size:12px;">5 分钟内有效。如果这不是你发起的，请忽略此邮件。</p>
          <hr style="border-color:#d4a574;">
          <p style="color:#aaa;font-size:10px;">— NaNaGi · 南志锦的 AI 女仆</p>
        </div>
      `,
    });

    console.log(`[Email] 验证码已发送: ${email} → ${code}`);
    return { success: true };
  } catch (err) {
    codeStore.delete(email.toLowerCase());
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`[Email] 发送失败: ${email} — ${msg}`);
    return { success: false, error: `邮件发送失败: ${msg}` };
  }
}
