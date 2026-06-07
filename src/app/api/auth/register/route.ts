// ============================================================
// NaNaGi 注册 API — P3-1
// 引导式表单 → personId + IWM Node 初始化
// 邮箱验证码 + 格式验证 + 一次性邮箱拦截 + 拼写纠错
// ============================================================

import { NextRequest, NextResponse } from "next/server";
import { nanoid } from "nanoid";
import bcrypt from "bcryptjs";
import { createToken, setAuthCookie } from "@/lib/auth";
import { putUser, putNode } from "@/lib/store";
import type { UserRecord } from "@/lib/leveldb";
import { initIWMFromForm, type RegisterForm } from "@/personality/iwm-init";
import { verifyCode } from "@/lib/email";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => ({}));

  // 验证必填字段
  const { email, name, password, identity, code } = body as RegisterForm & { code: string };
  if (!email || !name || !password || !identity || !code) {
    return NextResponse.json(
      { error: "请填写邮箱、名字、密码、身份和验证码" },
      { status: 400 }
    );
  }

  // 验证邮箱验证码
  if (!verifyCode(email, code)) {
    return NextResponse.json(
      { error: "验证码错误或已过期，请重新获取" },
      { status: 400 }
    );
  }

  // 邮箱格式验证
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return NextResponse.json(
      { error: "请输入有效的邮箱地址" },
      { status: 400 }
    );
  }

  // 一次性/临时邮箱拦截
  const domain = email.split("@")[1]?.toLowerCase() || "";
  const DISPOSABLE_DOMAINS = [
    "mailinator.com", "guerrillamail.com", "10minutemail.com", "tempmail.com",
    "throwaway.email", "yopmail.com", "sharklasers.com", "trashmail.com",
    "temp-mail.org", "fakeinbox.com", "guerrillamail.org", "mailnesia.com",
    "dispostable.com", "getnada.com", "tempinbox.com", "moakt.com",
  ];
  if (DISPOSABLE_DOMAINS.includes(domain)) {
    return NextResponse.json(
      { error: "请不要使用一次性邮箱注册" },
      { status: 400 }
    );
  }

  // 常见邮箱拼写错误提示
  const TYPO_FIXES: Record<string, string> = {
    "gmial.com": "gmail.com", "gmail.con": "gmail.com",
    "qq.con": "qq.com", "qq.cpm": "qq.com",
    "163.con": "163.com", "163.cpm": "163.com",
    "outlook.con": "outlook.com", "hotmail.con": "hotmail.com",
  };
  if (TYPO_FIXES[domain]) {
    const fixed = email.split("@")[0] + "@" + TYPO_FIXES[domain];
    return NextResponse.json(
      { error: `邮箱拼写可能有误，你是想说 ${fixed} 吗？` },
      { status: 400 }
    );
  }

  if (!["面试官", "普通用户"].includes(identity)) {
    return NextResponse.json(
      { error: "身份只能是'面试官'或'普通用户'" },
      { status: 400 }
    );
  }

  if (password.length < 4) {
    return NextResponse.json(
      { error: "密码至少需要 4 位" },
      { status: 400 }
    );
  }

  // 检查邮箱是否已注册
  const { getPersonIdByEmail, putEmailIndex } = await import("@/lib/store");
  const existing = await getPersonIdByEmail(email);
  if (existing) {
    return NextResponse.json(
      { error: "该邮箱已注册，请直接登录" },
      { status: 409 }
    );
  }

  // 生成 personId
  const personId = `guest-${nanoid(12)}`;
  const passwordHash = bcrypt.hashSync(password, 10);
  const now = new Date().toISOString();

  // 写用户记录
  const userRecord: UserRecord = {
    personId,
    email,
    name,
    passwordHash,
    role: identity === "面试官" ? "guest-iv" : "guest",
    identity,
    company: body.company || undefined,
    jobRole: body.jobRole || undefined,
    techInterests: body.techInterests || [],
    wantToKnow: body.wantToKnow || [],
    createdAt: now,
    lastLogin: now,
  };
  await putUser(userRecord);

  // 邮箱 → personId 索引
  await putEmailIndex(email, personId);

  // 初始化 IWM Node
  const iwmNode = initIWMFromForm(personId, body as RegisterForm);
  await putNode(iwmNode);

  // 签发 JWT
  const token = await createToken(personId, userRecord.role, name, identity);
  await setAuthCookie(token);

  console.log(`[Register] 新用户: ${name} (${identity}) → ${email} → ${personId}`);

  return NextResponse.json({
    success: true,
    personId,
    role: userRecord.role,
    name,
    identity,
    email,
  });
}
