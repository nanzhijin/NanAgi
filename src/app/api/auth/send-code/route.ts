// ============================================================
// NaNaGi 发送邮箱验证码 API — P3
// POST { email } → 发送 6 位验证码到邮箱
// ============================================================

import { NextRequest, NextResponse } from "next/server";
import { sendVerificationCode } from "@/lib/email";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => ({}));
  const { email } = body;

  if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return NextResponse.json(
      { error: "请输入有效的邮箱地址" },
      { status: 400 }
    );
  }

  const result = await sendVerificationCode(email);

  if (!result.success) {
    return NextResponse.json(
      { error: result.error || "验证码发送失败，请稍后重试" },
      { status: 500 }
    );
  }

  return NextResponse.json({ success: true });
}
