import { NextRequest, NextResponse } from "next/server";
import { listMemories, createMemory } from "@/lib/memory";
import type { MemoryMeta } from "@/lib/memory";
import { getAuthCookie, verifyToken } from "@/lib/auth";

/** 从请求中解析 role + personId */
async function getAuth(request: NextRequest) {
  const role = request.headers.get("x-nanagi-role") || "guest";
  const personId = request.headers.get("x-nanagi-person-id") || "guest";
  // 也支持从 cookie 直接读 (fallback)
  const token = await getAuthCookie();
  if (token) {
    const jwt = await verifyToken(token);
    if (jwt.valid) {
      return { role: jwt.role, personId: jwt.personId };
    }
  }
  return { role, personId };
}

export async function GET(request: NextRequest) {
  try {
    const { role, personId } = await getAuth(request);

    // admin: 读取所有历史记忆 / guest: 只读自己的
    const memories = role === "admin"
      ? await listMemories()
      : await (await import("@/lib/store")).listMemories(personId);

    const brief = memories.map(({ slug, meta, content }) => ({
      slug,
      meta,
      content: content?.slice(0, 300) || "", // guest 只给 300 字预览
      personId: "personId" in meta ? (meta as Record<string,unknown>).personId : undefined,
    }));
    return NextResponse.json(brief);
  } catch (err) {
    console.error("[Memory] GET error:", err);
    return NextResponse.json({ error: "读取记忆失败" }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const { role, personId } = await getAuth(request);
    const body = await request.json();
    const { meta, content } = body as { meta: MemoryMeta; content: string };
    if (!meta?.name || !meta?.description || !content) {
      return NextResponse.json(
        { error: "缺少必要字段 (name, description, content)" },
        { status: 400 }
      );
    }

    // guest: 写入 LevelDB / admin: 写入文件系统
    if (role !== "admin") {
      const store = await import("@/lib/store");
      await store.createMemory({
        slug: meta.name,
        personId,
        meta: {
          name: meta.name,
          description: meta.description,
          type: meta.type,
          tags: meta.tags || [],
          createdAt: new Date().toISOString(),
        },
        content,
        summary: meta.description,
        keywords: [],
      });
      return NextResponse.json({ success: true }, { status: 201 });
    }

    const entry = await createMemory(meta, content);
    return NextResponse.json(entry, { status: 201 });
  } catch (err) {
    console.error("[Memory] POST error:", err);
    return NextResponse.json({ error: "创建记忆失败" }, { status: 500 });
  }
}
