// ============================================================
// NaNaGi IWM Node 初始化器 — P3
// 注册表单 → IWM Node 冷启动
// 身份类型决定初始 traits 基线
// ============================================================

import type { IWMNode } from "@/lib/leveldb";

export interface RegisterForm {
  email: string;
  name: string;
  password: string;
  identity: "面试官" | "普通用户";
  company?: string;
  jobRole?: string;
  techInterests?: string[];
  wantToKnow?: string[];
}

/** 身份 → 初始 traits 基线 */
const IDENTITY_BASELINE: Record<string, IWMNode["traits"]> = {
  "面试官":   { safety: 0.60, intimacy: 0.10, care: 0.50, respect: 0.55, reliability: 0.50, understanding: 0.30 },
  "普通用户": { safety: 0.50, intimacy: 0.10, care: 0.50, respect: 0.50, reliability: 0.50, understanding: 0.30 },
};

export function initIWMFromForm(personId: string, form: RegisterForm): IWMNode {
  const traits = IDENTITY_BASELINE[form.identity] || IDENTITY_BASELINE["普通用户"];

  const knownFacts: string[] = [];
  if (form.identity === "面试官" && form.company) {
    knownFacts.push(`${form.company}面试官`);
  }
  if (form.jobRole) {
    knownFacts.push(`招聘${form.jobRole}`);
  }
  if (form.techInterests?.length) {
    knownFacts.push(`关注${form.techInterests.join("、")}`);
  }
  if (form.wantToKnow?.length) {
    knownFacts.push(`想了解${form.wantToKnow.join("、")}`);
  }

  const topicInterests = [...(form.techInterests || []), ...(form.wantToKnow || [])];

  return {
    personId,
    name: form.name,
    role: form.identity === "面试官" ? "guest-iv" : "guest",
    identity: form.identity,
    traits: { ...traits },
    knownFacts,
    topicInterests,
    company: form.company || undefined,
    jobRole: form.jobRole || undefined,
    firstMet: new Date().toISOString(),
    lastTalk: new Date().toISOString(),
    totalTurns: 0,
    historyDensity: 0.0,
  };
}
