// ============================================================
// NaNaGi Agent 工具 — save_memory
// 主动保存记忆 — NaNaGi 判断"这个值得记"时自主调用
// admin → 文件系统 / guest → LevelDB
// ============================================================

import { registerTool } from "../registry";
import type { ToolResult, AgentContext } from "../types";
import { createMemory as fsCreateMemory } from "@/lib/memory";
import type { MemoryType } from "@/lib/memory";

registerTool({
  definition: {
    name: "save_memory",
    description:
      "记住访客/主人的重要信息。当客人分享个人信息、表达偏好、提出反馈、或对话中出现任何值得后续参考的内容时，主动调用此工具记录。不需要客人说'记住'。",
    input_schema: {
      type: "object",
      properties: {
        memory_type: {
          type: "string",
          enum: ["user", "project", "impression", "feedback"],
          description:
            "记忆类型: user=访客身份信息, project=项目相关讨论, impression=印象/偏好, feedback=反馈/建议",
        },
        description: {
          type: "string",
          description:
            "一行简短摘要，描述这条记忆的内容（例: '面试官来自字节跳动，关注推荐系统'）",
        },
        content: {
          type: "string",
          description:
            "要记住的完整内容。用自然语言写，包含关键信息和上下文。",
        },
        tags: {
          type: "string",
          description: "逗号分隔的标签（可选，例: '面试,字节跳动,推荐系统'）",
        },
      },
      required: ["memory_type", "description", "content"],
    },
  },

  async execute(args, ctx: AgentContext): Promise<ToolResult> {
    const memType = args.memory_type as string;
    const desc = args.description as string;
    const memContent = args.content as string;
    const tags = args.tags
      ? (args.tags as string).split(",").map((s) => s.trim()).filter(Boolean)
      : [];
    const ts = new Date().toISOString();

    try {
      // admin → 文件系统 / guest → LevelDB
      if (ctx.role !== "admin") {
        const store = await import("@/lib/store");
        await store.createMemory({
          slug: `mem-${Date.now()}`,
          personId: ctx.personId,
          meta: {
            name: `mem-${Date.now()}`,
            description: desc,
            type: memType as MemoryType,
            tags: tags.length > 0 ? tags : [],
            createdAt: ts,
          },
          content: memContent,
          summary: desc,
          keywords: tags,
        });
      } else {
        const slug = `mem-${Date.now()}`;
        await fsCreateMemory(
          {
            name: slug,
            description: desc,
            type: memType as MemoryType,
            tags: tags.length > 0 ? tags : undefined,
          },
          memContent
        );
      }

      return {
        tool_call_id: "",
        content: `记忆已保存: 「${desc}」`,
      };
    } catch (err) {
      return {
        tool_call_id: "",
        content: `记忆保存失败: ${err instanceof Error ? err.message : String(err)}。`,
        is_error: true,
      };
    }
  },
});
