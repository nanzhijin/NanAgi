// ============================================================
// NaNaGi 注册 API — P3-1
// 引导式表单 → personId + IWM Node 初始化
// ============================================================

import { NextRequest, NextResponse } from "next/server";
import { nanoid } from "nanoid";
import bcrypt from "bcryptjs";
import { createToken, setAuthCookie } from "@/lib/auth";
import { putUser, putNode } from "@/lib/store";
import type { UserRecord } from "@/lib/leveldb";
import { initIWMFromForm, type RegisterForm } from "@/personality/iwm-init";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => ({}));

  // 验证必填字段
  const { name, password, identity } = body as RegisterForm;
  if (!name || !password || !identity) {
    return NextResponse.json(
      { error: "请填写名字、密码和身份" },
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

  // 生成 personId
  const personId = `guest-${nanoid(12)}`;
  const passwordHash = bcrypt.hashSync(password, 10);
  const now = new Date().toISOString();

  // 写用户记录
  const userRecord: UserRecord = {
    personId,
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

  // 初始化 IWM Node
  const iwmNode = initIWMFromForm(personId, body as RegisterForm);
  await putNode(iwmNode);

  // 签发 JWT
  const token = await createToken(personId, userRecord.role, name, identity);
  await setAuthCookie(token);

  console.log(`[Register] 新用户: ${name} (${identity}) → ${personId}`);

  return NextResponse.json({
    success: true,
    personId,
    role: userRecord.role,
    name,
    identity,
  });
}
