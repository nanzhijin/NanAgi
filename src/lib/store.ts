// ============================================================
// NaNaGi 统一数据访问层 — P2-1
// 调用方不感知底层是文件系统还是 LevelDB
// 内部路由: admin(nanzhijin) → 文件系统, guest → LevelDB
// ============================================================

import fs from "fs/promises";
import path from "path";
import { existsSync } from "fs";
import {
  getIWMNode as levelGetIWM,
  putIWMNode as levelPutIWM,
  putMemoryRecord as levelPutMemory,
  listMemoryRecords as levelListMemories,
  putCellRecord as levelPutCell,
  getCellRecord as levelGetCell,
  listCellRecords as levelListCells,
  putLastSummary as levelPutLastSummary,
  getLastSummary as levelGetLastSummary,
  putFeedbackRecord as levelPutFeedback,
  getFeedbackRecord as levelGetFeedback,
  putUserRecord as levelPutUser,
  getUserRecord as levelGetUser,
  listGuestNodesByActivity,
  type IWMNode,
  type MemoryRecord,
  type CellRecord,
  type FeedbackRecord,
  type UserRecord,
} from "./leveldb";

// ==================== 目录 ====================

const DATA_DIR = path.join(process.cwd(), "data");
const ADMIN_DIR = path.join(DATA_DIR, "admin");
const ADMIN_IWM_FILE = path.join(ADMIN_DIR, "nanzhijin-iwm.json");
const ADMIN_AUTH_FILE = path.join(ADMIN_DIR, "nanzhijin.json");

async function ensureDir(dir: string): Promise<void> {
  if (!existsSync(dir)) {
    await fs.mkdir(dir, { recursive: true });
  }
}

// ==================== IWM Node ====================

export async function getNode(personId: string): Promise<IWMNode | null> {
  if (personId === "nanzhijin") {
    try {
      return JSON.parse(await fs.readFile(ADMIN_IWM_FILE, "utf-8"));
    } catch {
      // 首次使用 → 初始化空白节点
      const initNode: IWMNode = {
        personId: "nanzhijin", name: "南志锦", role: "admin", identity: "主人",
        traits: { safety: 0.5, intimacy: 0.1, care: 0.5, respect: 0.5, reliability: 0.5, understanding: 0.3 },
        knownFacts: [], topicInterests: [],
        firstMet: new Date().toISOString(), lastTalk: new Date().toISOString(),
        totalTurns: 0, historyDensity: 0.0,
      };
      await putNode(initNode);
      return initNode;
    }
  }
  return levelGetIWM(personId);
}

export async function putNode(node: IWMNode): Promise<void> {
  if (node.personId === "nanzhijin") {
    await ensureDir(ADMIN_DIR);
    await fs.writeFile(ADMIN_IWM_FILE, JSON.stringify(node, null, 2), "utf-8");
    return;
  }
  await levelPutIWM(node);
}

// ==================== Memory ====================

export async function createMemory(record: MemoryRecord): Promise<void> {
  if (record.personId === "nanzhijin") {
    const dir = path.join(ADMIN_DIR, "memories");
    await ensureDir(dir);
    const slug = record.slug || `mem-${Date.now()}`;
    const content = `---\ndescription: ${record.meta.description}\ntype: ${record.meta.type}\ncreatedAt: ${record.meta.createdAt}\n---\n\n${record.content}`;
    await fs.writeFile(path.join(dir, `${slug}.md`), content, "utf-8");
    return;
  }
  await levelPutMemory(record);
}

export async function listMemories(personId: string): Promise<MemoryRecord[]> {
  if (personId === "nanzhijin") {
    // 保持现有文件系统逻辑 — lib/memory.ts 处理
    // store.ts 不重复实现, 调现有 API
    const { listMemories: fsListMemories } = await import("./memory");
    const entries = await fsListMemories();
    return entries.map((e) => ({
      slug: e.slug,
      personId: "nanzhijin",
      meta: {
        name: e.meta.name,
        description: e.meta.description,
        type: e.meta.type,
        tags: e.meta.tags || [],
        createdAt: e.meta.createdAt || e.updatedAt,
      },
      content: e.content,
      summary: "",
      keywords: [],
    }));
  }
  return levelListMemories(personId);
}

// ==================== User ====================

export async function putUser(record: UserRecord): Promise<void> {
  await levelPutUser(record);
}

export async function getUser(personId: string): Promise<UserRecord | null> {
  if (personId === "nanzhijin") {
    try {
      return JSON.parse(await fs.readFile(ADMIN_AUTH_FILE, "utf-8"));
    } catch {
      return null;
    }
  }
  return levelGetUser(personId);
}

// ==================== Cell ====================

export async function putCell(record: CellRecord): Promise<void> {
  await levelPutCell(record);
}

export async function getCell(personId: string, cellId: string): Promise<CellRecord | null> {
  return levelGetCell(personId, cellId);
}

export async function listCells(personId: string): Promise<CellRecord[]> {
  return levelListCells(personId);
}

export async function putLastSummary(personId: string, summary: string): Promise<void> {
  await levelPutLastSummary(personId, summary);
}

export async function getLastSummary(personId: string): Promise<string | null> {
  return levelGetLastSummary(personId);
}

// ==================== Feedback ====================

export async function putFeedback(record: FeedbackRecord): Promise<void> {
  await levelPutFeedback(record);
}

export async function getFeedback(personId: string): Promise<FeedbackRecord | null> {
  return levelGetFeedback(personId);
}

// ==================== Admin Auth ====================

export async function putAdminAuth(auth: { personId: string; passwordHash: string; role: string }): Promise<void> {
  await ensureDir(ADMIN_DIR);
  await fs.writeFile(ADMIN_AUTH_FILE, JSON.stringify(auth, null, 2), "utf-8");
}

export async function getAdminAuth(): Promise<{ personId: string; passwordHash: string; role: string } | null> {
  try {
    return JSON.parse(await fs.readFile(ADMIN_AUTH_FILE, "utf-8"));
  } catch {
    return null;
  }
}

// ==================== Email 索引 ====================

import { putEmailIndex as levelPutEmail, getPersonIdByEmail as levelGetEmail } from "./leveldb";

export { putEmailIndex, getPersonIdByEmail } from "./leveldb";

// ==================== Guest 查询 ====================

export { listGuestNodesByActivity };
