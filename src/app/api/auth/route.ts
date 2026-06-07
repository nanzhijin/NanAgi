// ============================================================
// NaNaGi 登录 API — P3-2
// admin: 密码登录 (保持兼容)
// guest: personId + 密码 → LevelDB 验证
// ============================================================

import { NextRequest, NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import {
  createToken,
  setAuthCookie,
  verifyToken,
  verifyPassword,
  getAuthCookie,
} from "@/lib/auth";
import { getUser, getPersonIdByEmail } from "@/lib/store";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => ({}));
  const { personId, email, password } = body;

  if (!password) {
    return NextResponse.json({ error: "请输入密码" }, { status: 400 });
  }

  // —— Admin 登录: 仅密码, 无 email ——
  // ⚠️ 只要带了 email 就不是 admin 登录, 防止 guest 密码与 admin 密码冲突
  if (!email && !personId) {
    const result = verifyPassword(password);
    if (result.valid) {
      const token = await createToken("nanzhijin", "admin", "南志锦", "主人");
      await setAuthCookie(token);
      return NextResponse.json({
        success: true,
        personId: "nanzhijin",
        role: "admin",
        name: "南志锦",
        identity: "主人",
      });
    }
    return NextResponse.json({ error: "密码错误，请重试" }, { status: 401 });
  }

  // —— Guest 登录: email + password ——
  // 如果提供了 email，先通过 email 查找 personId
  let lookupId = personId;
  if (!lookupId && email) {
    lookupId = await getPersonIdByEmail(email);
    if (!lookupId) {
      return NextResponse.json(
        { error: "该邮箱未注册，请先注册" },
        { status: 401 }
      );
    }
  }

  const user = await getUser(lookupId || "");
  if (!user) {
    return NextResponse.json(
      { error: "用户不存在，请检查你的邮箱或专属 ID 是否正确" },
      { status: 401 }
    );
  }

  const valid = bcrypt.compareSync(password, user.passwordHash);
  if (!valid) {
    return NextResponse.json({ error: "密码错误，请重试" }, { status: 401 });
  }

  // 更新最后登录时间 (最佳努力)
  user.lastLogin = new Date().toISOString();
  const { putUser } = await import("@/lib/store");
  await putUser(user).catch(() => {});

  const token = await createToken(
    user.personId,
    user.role,
    user.name,
    user.identity
  );
  await setAuthCookie(token);

  return NextResponse.json({
    success: true,
    personId: user.personId,
    role: user.role,
    name: user.name,
    identity: user.identity,
  });
}

export async function GET() {
  const token = await getAuthCookie();
  if (!token) {
    return NextResponse.json({ authenticated: false, role: null });
  }
  const jwt = await verifyToken(token);
  if (!jwt.valid) {
    return NextResponse.json({ authenticated: false, role: null });
  }

  // 验证用户是否真实存在 (admin 跳过)
  if (jwt.role !== "admin") {
    const user = await getUser(jwt.personId);
    if (!user) {
      // 用户已被删除 → 清 cookie + 返回未认证
      const { cookies: getCookies } = await import("next/headers");
      (await getCookies()).delete("nanagi_token");
      return NextResponse.json({ authenticated: false, role: null });
    }
  }

  return NextResponse.json({
    authenticated: true,
    role: jwt.role,
    personId: jwt.personId,
    name: jwt.name,
    identity: jwt.identity,
  });
}
