import { NextRequest, NextResponse } from "next/server";
import { getMemory, deleteMemory } from "@/lib/memory";
import { getAuthCookie, verifyToken } from "@/lib/auth";

async function getRole(request: NextRequest): Promise<string> {
  const role = request.headers.get("x-nanagi-role") || "guest";
  const token = await getAuthCookie();
  if (token) {
    const jwt = await verifyToken(token);
    if (jwt.valid) return jwt.role;
  }
  return role;
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ slug: string }> }
) {
  try {
    const role = await getRole(request);
    if (role !== "admin") {
      return NextResponse.json({ error: "仅管理员可查看" }, { status: 403 });
    }
    const { slug } = await params;
    const entry = await getMemory(slug);
    if (!entry) {
      return NextResponse.json({ error: "记忆未找到" }, { status: 404 });
    }
    return NextResponse.json(entry);
  } catch (err) {
    console.error("[Memory] GET slug error:", err);
    return NextResponse.json({ error: "读取记忆失败" }, { status: 500 });
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ slug: string }> }
) {
  try {
    const role = await getRole(request);
    if (role !== "admin") {
      return NextResponse.json({ error: "仅管理员可删除" }, { status: 403 });
    }
    const { slug } = await params;
    const ok = await deleteMemory(slug);
    if (!ok) {
      return NextResponse.json({ error: "记忆未找到" }, { status: 404 });
    }
    return NextResponse.json({ success: true });
  } catch (err) {
    console.error("[Memory] DELETE error:", err);
    return NextResponse.json({ error: "删除记忆失败" }, { status: 500 });
  }
}
