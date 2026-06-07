// ============================================================
// NaNaGi Chat API — P1-7 (重构: 644→~90行)
// 薄层 handler: 认证 → 环境感知 → AgentContext → agentLoop
// ============================================================

import { NextRequest } from "next/server";
import { verifyToken, getAuthCookie } from "@/lib/auth";
import { agentLoop } from "@/agent/loop";
import { buildSystemPrompt } from "@/agent/prompts";
import { guestConfig } from "@/personality/configs/guest";
import { adminConfig } from "@/personality/configs/admin";
import { getAmbient } from "@/lib/ambient";
import { listMemories, createMemory, type MemoryType } from "@/lib/memory";
import { getNode, putNode, createMemory as storeCreateMemory } from "@/lib/store";
import type { MemoryRecord, IWMNode } from "@/lib/leveldb";
import type { AgentContext, AgentMessage } from "@/agent/types";

// 动态导入工具 — 触发 registerTool()
import "@/agent/tools";

// ==================== 记忆上下文 (保留现有逻辑) ====================

async function buildMemoryContext(personId: string, role: string): Promise<string> {
  try {
    // admin: 读取所有历史记忆 (文件系统 + LevelDB)
    // guest: 只读取该用户自己的记忆 (LevelDB, 按 personId 过滤)
    // admin: 读取所有历史记忆 / guest: 只读自己的
    const rawMemories = role === "admin"
      ? await listMemories()
      : await (await import("@/lib/store")).listMemories(personId);

    const memories = rawMemories.map((m) => ({
      meta: { type: m.meta.type, description: m.meta.description },
      content: m.content,
    }));

    if (memories.length === 0) return "";

    const labels: Record<string, string> = {
      user: "### 👤 访客档案",
      project: "### 📁 项目记忆",
      impression: "### 💭 印象笔记",
      feedback: "### 📝 反馈记录",
    };

    const groups: Record<string, string[]> = {};
    for (const m of memories) {
      const group = (groups[m.meta.type] ??= []);
      const snippet =
        m.content.length > 300 ? m.content.slice(0, 300) + "..." : m.content;
      group.push(`- **${m.meta.description}**: ${snippet}`);
    }

    let ctx = "\n\n---\n## 📚 已有记忆\n\n";
    for (const [type, items] of Object.entries(groups)) {
      ctx += `${labels[type] || type}\n${items.join("\n")}\n\n`;
    }
    return ctx;
  } catch {
    return "";
  }
}

// ==================== 框架层记忆拦截 (保留) ====================

const MEMORY_TRIGGERS = /(?:记忆|记住|记录|记一下|memory|备忘|存档)/i;

function checkMemoryTrigger(text: string): boolean {
  return MEMORY_TRIGGERS.test(text);
}

function resolveMemoryType(text: string): MemoryType {
  if (/bug|问题|错误|故障|失败/.test(text)) return "feedback";
  if (/项目|CNN|GNN|音乐|模型|识别|推荐/.test(text)) return "project";
  if (/我是|面试|公司|职位|技术栈/.test(text)) return "user";
  return "impression";
}

function extractMemoryDesc(text: string): string {
  const cleaned = text
    .replace(/娜娜吉[，,]?\s*/g, "")
    .replace(/(?:记忆|记住|记录|记一下|memory|备忘|存档)[，,。.]?\s*/gi, "")
    .trim();
  const firstSegment = cleaned.split(/[，,。.！!？?\n]/)[0].trim();
  return firstSegment.slice(0, 40) || "未命名记忆";
}

// ==================== POST Handler ====================

export async function POST(request: NextRequest) {
  // 1. 认证
  const token = await getAuthCookie();
  if (!token) return new Response("Unauthorized", { status: 401 });

  const jwt = await verifyToken(token);
  if (!jwt.valid) return new Response("Token expired", { status: 401 });

  // P3: personId + name + identity 直接从 JWT 读取
  const { personId, role, name, identity } = jwt;

  // 2. 环境感知
  const ambient = await getAmbient(request);

  // 3. 构建 AgentContext
  const body = await request.json().catch(() => ({}));
  const { messages = [], project } = body;

  const memoryContext = await buildMemoryContext(personId, role);

  const ctx: AgentContext = {
    personId,
    role: role as AgentContext["role"],
    name,
    identity,
    config: role === "admin" ? adminConfig : guestConfig,
    project,
    ambient,
    memoryContext,
  };

  // 4. 格式化消息
  const agentMessages: AgentMessage[] = messages.map(
    (m: { role: string; content: string }) => ({
      role: m.role === "agent" ? "assistant" : "user",
      content: m.content,
    })
  );

  // 5. 框架层记忆拦截: 检测最后一条用户消息
  const lastUserMsg = [...messages]
    .reverse()
    .find((m: { role: string }) => m.role === "user") as
    | { role: string; content: string }
    | undefined;
  const shouldMemorize =
    lastUserMsg && checkMemoryTrigger(lastUserMsg.content);

  // 6. 启动 Agent Loop → SSE stream
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      try {
        await agentLoop(controller, ctx, agentMessages);
      } catch (err) {
        console.error("[Chat] Agent loop error:", err);
        const fallback =
          role === "admin"
            ? "主人...我的大脑好像卡住了，能等一下吗？"
            : "抱歉呢，AI 服务暂时不可用，请稍后重试。";
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({ type: "text", content: fallback })}\n\n`)
        );
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      }

      // ==================== Step 10: 后处理 ====================

      // —— IWM Node 持久化 ——
      try {
        let iwmNode = await getNode(personId);
        if (!iwmNode) {
          // 首次接触 → 初始化节点
          const isAdmin = personId === "nanzhijin";
          iwmNode = {
            personId, name, role: role as IWMNode["role"], identity,
            traits: isAdmin
              ? { safety: 0.85, intimacy: 0.50, care: 0.80, respect: 0.75, reliability: 0.70, understanding: 0.55 }
              : { safety: 0.5, intimacy: 0.1, care: 0.5, respect: 0.5, reliability: 0.5, understanding: 0.3 },
            knownFacts: [],
            topicInterests: [],
            firstMet: new Date().toISOString(), lastTalk: new Date().toISOString(),
            totalTurns: 0, historyDensity: 0.0,
          };
          console.log("[Store] 初始化 IWM Node:", personId, isAdmin ? "(admin基线)" : "(guest基线)");
        }
        iwmNode.totalTurns++;
        iwmNode.lastTalk = new Date().toISOString();
        iwmNode.historyDensity = Math.min(1.0, iwmNode.totalTurns / 100);
        await putNode(iwmNode);
      } catch (err) {
        console.error("[Store] IWM 持久化失败:", err);
      }

      // —— 框架层记忆拦截 (统一走 store) ——
      if (shouldMemorize && lastUserMsg) {
        try {
          const desc = extractMemoryDesc(lastUserMsg.content);
          const memType = resolveMemoryType(lastUserMsg.content);
          const slug = `mem-${Date.now()}`;
          const ts = new Date().toISOString();

          await storeCreateMemory({
            slug,
            personId,
            meta: {
              name: slug,
              description: desc,
              type: memType,
              tags: [],
              createdAt: ts,
            },
            content: `## 用户消息\n${lastUserMsg.content}\n\n## 对话摘要\n记忆触发词已记录`,
            summary: desc,
            keywords: [],
          });
          console.log("[Store] 记忆已保存:", desc, personId === "nanzhijin" ? "(fs)" : "(LevelDB)");
        } catch (err) {
          console.error("[Store] 记忆保存失败:", err);
        }
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
