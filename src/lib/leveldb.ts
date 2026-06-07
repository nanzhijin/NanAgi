// ============================================================
// NaNaGi K-V 存储引擎 — P2-2 (v2: 用户子目录隔离)
// 文件系统 K-V, 接口可替换为真实 LevelDB
// 六表命名空间: user / iwm / mem / emo / conv / feedback
// 目录结构:
//   data/leveldb/{personId}/   ← 每人独立目录
//   data/leveldb/_index/       ← 全局索引
// ============================================================

import fs from "fs/promises";
import path from "path";
import { existsSync } from "fs";

const DB_DIR = path.join(process.cwd(), "data", "leveldb");
const INDEX_DIR = path.join(DB_DIR, "_index");

async function ensureDir(dir: string): Promise<void> {
  if (!existsSync(dir)) await fs.mkdir(dir, { recursive: true });
}

async function userDir(personId: string): Promise<string> {
  const dir = path.join(DB_DIR, personId);
  await ensureDir(dir);
  return dir;
}

async function readJSON<T>(filePath: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(filePath, "utf-8")) as T;
  } catch {
    return null;
  }
}

async function writeJSON(filePath: string, data: unknown): Promise<void> {
  await fs.writeFile(filePath, JSON.stringify(data, null, 2), "utf-8");
}

// ==================== 全局扫描 ====================

/** 列出所有 guest 用户 personId */
async function listPersonIds(): Promise<string[]> {
  await ensureDir(DB_DIR);
  const entries = await fs.readdir(DB_DIR, { withFileTypes: true });
  return entries
    .filter((e) => e.isDirectory() && e.name.startsWith("guest-"))
    .map((e) => e.name);
}

// ==================== 通用 Key-Value ====================

export async function dbPut(key: string, value: Record<string, unknown>): Promise<void> {
  await ensureDir(DB_DIR);
  await writeJSON(path.join(DB_DIR, `${key}.json`), value);
}

export async function dbGet<T = Record<string, unknown>>(key: string): Promise<T | null> {
  return readJSON<T>(path.join(DB_DIR, `${key}.json`));
}

export async function dbDelete(key: string): Promise<void> {
  try { await fs.unlink(path.join(DB_DIR, `${key}.json`)); } catch { /* ignore */ }
}

// ==================== 用户 — Table 1 ====================

export interface UserRecord {
  personId: string;
  email: string;
  name: string;
  passwordHash: string;
  role: "guest-iv" | "guest";
  identity: "面试官" | "普通用户";
  company?: string;
  jobRole?: string;
  techInterests?: string[];
  wantToKnow?: string[];
  createdAt: string;
  lastLogin: string;
}

export async function putUserRecord(record: UserRecord): Promise<void> {
  const dir = await userDir(record.personId);
  await writeJSON(path.join(dir, "user.json"), record);
}

export async function getUserRecord(personId: string): Promise<UserRecord | null> {
  return readJSON<UserRecord>(path.join(DB_DIR, personId, "user.json"));
}

// ==================== Email 索引 ====================

async function emailIndexDir(): Promise<string> {
  const dir = path.join(INDEX_DIR, "email");
  await ensureDir(dir);
  return dir;
}

function emailKey(email: string): string {
  return email.toLowerCase().replace(/[@.]/g, "_");
}

export async function putEmailIndex(email: string, personId: string): Promise<void> {
  const dir = await emailIndexDir();
  await writeJSON(path.join(dir, `${emailKey(email)}.json`), { email, personId });
}

export async function getPersonIdByEmail(email: string): Promise<string | null> {
  const data = await readJSON<{ personId: string }>(
    path.join(INDEX_DIR, "email", `${emailKey(email)}.json`)
  );
  return data?.personId || null;
}

// ==================== IWM Node — Table 2 ====================

export interface IWMNode {
  personId: string;
  name: string;
  role: "admin" | "guest-iv" | "guest";
  identity: string;
  traits: {
    safety: number; intimacy: number; care: number;
    respect: number; reliability: number; understanding: number;
  };
  knownFacts: string[];
  topicInterests: string[];
  company?: string;
  jobRole?: string;
  firstMet: string;
  lastTalk: string;
  totalTurns: number;
  historyDensity: number;
}

export async function putIWMNode(node: IWMNode): Promise<void> {
  const dir = await userDir(node.personId);
  await writeJSON(path.join(dir, "iwm.json"), node);
}

export async function getIWMNode(personId: string): Promise<IWMNode | null> {
  return readJSON<IWMNode>(path.join(DB_DIR, personId, "iwm.json"));
}

// ==================== 记忆 — Table 3 ====================

export interface MemoryRecord {
  slug: string;
  personId: string;
  meta: {
    name: string;
    description: string;
    type: "user" | "project" | "impression" | "feedback";
    tags: string[];
    createdAt: string;
  };
  content: string;
  summary: string;
  keywords: string[];
}

export async function putMemoryRecord(record: MemoryRecord): Promise<void> {
  const dir = await userDir(record.personId);
  const ts = record.meta.createdAt || new Date().toISOString();
  await ensureDir(path.join(dir, "memories"));
  const safeTs = ts.replace(/[:]/g, "-");
  await writeJSON(path.join(dir, "memories", `${safeTs}.json`), record);
}

export async function listMemoryRecords(personId: string): Promise<MemoryRecord[]> {
  const memDir = path.join(DB_DIR, personId, "memories");
  if (!existsSync(memDir)) return [];
  const files = await fs.readdir(memDir);
  const records: MemoryRecord[] = [];
  for (const file of files) {
    if (!file.endsWith(".json")) continue;
    const r = await readJSON<MemoryRecord>(path.join(memDir, file));
    if (r) records.push(r);
  }
  records.sort((a, b) => (b.meta.createdAt || "").localeCompare(a.meta.createdAt || ""));
  return records;
}

// ==================== 情绪轨迹 — Table 4 ====================

export interface EmotionEntry {
  timestamp: string;
  personId: string;
  roundNumber: number;
  before: Record<string, number>;
  delta: Record<string, number>;
  trigger: { type: string; summary: string };
  ambientMood: Record<string, number>;
}

export async function putEmotionEntry(entry: EmotionEntry): Promise<void> {
  const dir = await userDir(entry.personId);
  await ensureDir(path.join(dir, "emotions"));
  await writeJSON(path.join(dir, "emotions", `${entry.timestamp.replace(/[:]/g, "-")}.json`), entry);
}

// ==================== Cell 会话 — Table 5 ====================

export interface CellRecord {
  cellId: string;
  personId: string;
  title: string;
  createdAt: string;
  lastMessageAt: string;
  messageCount: number;
  messages: { id: string; role: string; content: string; timestamp: string }[];
  toolCallsUsed?: string[];
  summary?: string;
}

export async function putCellRecord(record: CellRecord): Promise<void> {
  const dir = await userDir(record.personId);
  await ensureDir(path.join(dir, "cells"));
  await writeJSON(path.join(dir, "cells", `${record.cellId}.json`), record);
  // 维护 cell 索引
  const idx = await readJSON<string[]>(path.join(dir, "cells", "_index.json")) || [];
  if (!idx.includes(record.cellId)) {
    idx.push(record.cellId);
    await writeJSON(path.join(dir, "cells", "_index.json"), idx);
  }
}

export async function getCellRecord(personId: string, cellId: string): Promise<CellRecord | null> {
  return readJSON<CellRecord>(path.join(DB_DIR, personId, "cells", `${cellId}.json`));
}

export async function listCellRecords(personId: string): Promise<CellRecord[]> {
  const idx = await readJSON<string[]>(path.join(DB_DIR, personId, "cells", "_index.json")) || [];
  const cells: CellRecord[] = [];
  for (const cid of idx) {
    const cell = await readJSON<CellRecord>(path.join(DB_DIR, personId, "cells", `${cid}.json`));
    if (cell) cells.push(cell);
  }
  return cells;
}

export async function putLastSummary(personId: string, summary: string): Promise<void> {
  const dir = await userDir(personId);
  await writeJSON(path.join(dir, "last-summary.json"), { summary, updatedAt: new Date().toISOString() });
}

export async function getLastSummary(personId: string): Promise<string | null> {
  const data = await readJSON<{ summary: string }>(path.join(DB_DIR, personId, "last-summary.json"));
  return data?.summary || null;
}

// ==================== 面试反馈 — Table 6 ====================

export interface FeedbackRecord {
  personId: string;
  name: string;
  records: {
    timestamp: string;
    sessionId: string;
    cellId: string;
    company: string;
    role: string;
    impression: string;
    projectInterest: string[];
    quotes: string[];
  }[];
  summary: string;
}

export async function putFeedbackRecord(record: FeedbackRecord): Promise<void> {
  const dir = await userDir(record.personId);
  await writeJSON(path.join(dir, "feedback.json"), record);
}

export async function getFeedbackRecord(personId: string): Promise<FeedbackRecord | null> {
  return readJSON<FeedbackRecord>(path.join(DB_DIR, personId, "feedback.json"));
}

// ==================== 全局查询 ====================

/** 按角色和最近活跃时间筛选用户 */
export async function listGuestNodesByActivity(
  daysBack: number,
  identity?: string
): Promise<{ personId: string; name: string; identity: string; lastTalk: string; totalTurns: number }[]> {
  const cutoff = Date.now() - daysBack * 24 * 60 * 60 * 1000;
  const results: { personId: string; name: string; identity: string; lastTalk: string; totalTurns: number }[] = [];
  const ids = await listPersonIds();

  for (const personId of ids) {
    const node = await getIWMNode(personId);
    if (!node) continue;
    if (identity && node.identity !== identity) continue;
    if (new Date(node.lastTalk).getTime() < cutoff) continue;
    results.push({
      personId: node.personId, name: node.name, identity: node.identity,
      lastTalk: node.lastTalk, totalTurns: node.totalTurns,
    });
  }

  results.sort((a, b) => new Date(b.lastTalk).getTime() - new Date(a.lastTalk).getTime());
  return results;
}
