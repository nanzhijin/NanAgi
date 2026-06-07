# NaNaGi（ななぎ）

**南志锦的个人 AI 作品集网站。一个基于社交图 (Social Graph) 的关系型 Agent——不是帮你做事的工具，而是知道你是谁、记得你什么样、对不同的人不同对待的数字人格系统。**

---

## 目录

1. [设计哲学](#设计哲学)
2. [核心架构：社交图 + 三层心理](#核心架构社交图--三层心理)
3. [三角色分层系统](#三角色分层系统)
4. [双通道分化](#双通道分化)
5. [双图架构](#双图架构)
6. [环境感知：AmbientContext](#环境感知ambientcontext)
7. [注册即 Cold Start](#注册即-cold-start)
8. [Cell 隔离记忆架构](#cell-隔离记忆架构)
9. [完整对话数据流](#完整对话数据流)
10. [三层分级存储架构](#三层分级存储架构)
11. [LevelDB 六表 Schema](#leveldb-六表-schema)
12. [GNN 概念映射](#gnn-概念映射)
13. [待研究：IWM 更新触发条件](#待研究iwm-更新触发条件)
14. [学术支撑](#学术支撑)
15. [当前状态与路线图](#当前状态与路线图)
16. [字节 JD 差距审计](#字节-jd-差距审计)
17. [历史规划存档 (Block×P & P0-P4)](#历史规划存档)
18. [实施计划 v5.1 (Phase 1-8, 当前)](#实施计划-v51)
19. [完整文件清单](#完整文件清单)
20. [技术栈](#技术栈)
18. [本地运行](#本地运行)
19. [面试话术](#面试话术)

---

## 设计哲学

### 传统 Agent vs NaNaGi

```
工具型 Agent:  帮你完成任务。评价标准: 做成了没有？
关系型 Agent:  维持关系。评价标准: 她记得我吗？她对我跟对别人不一样吗？
```

市面上 99% 的 Agent 项目（LangChain、AutoGPT、CrewAI、Open-AGC）在解决同一个问题：**怎么让 LLM 更好地完成任务**。多 Agent 协作是为了拆解复杂任务，ReAct 循环是为了调对工具，RAG 是为了查对信息。

NaNaGi 解决一个完全不同的问题：**怎么让 LLM 有持续的关系记忆和社交情境感知**。她的核心不是"完成你交代的事"，而是"知道你是谁、记得你什么样、对不同的人不同对待"。

把 Agent 从"会做事的工具"升级为"会认人的存在"，是 NaNaGi 最根本的设计目标。

### 为什么不用 LangChain

LangChain 的抽象是为"任务链"设计的——Chain、Agent、Tool 三者围绕 task completion 组织。NaNaGi 的核心抽象是"人"——Self-Node、IWM Node、Social Graph ——围绕 relationship maintenance 组织。

**每引入一个框架抽象，都要问：这个抽象是为"做事"服务的，还是为"认人"服务的？** 前者，不用。自建。

### 设计红线

- ❌ 不用 LangChain/LangGraph — 自建 Agent loop 更干净
- ❌ 不用 SQLite — 文件系统可审计的哲学保留给 NaNaGi 本体
- ❌ 不做文件操作/代码执行/终端命令 — Bounded-Domain Agent 的边界
- ❌ 不引入超过 30 行的机制 — 每个机制必须能肉眼验证
- ✅ NaNaGi 的代码是女儿的皮肤 — 不能脏

---

## 核心架构：社交图 + 三层心理

### 总览

NaNaGi 的数字人格由四个子系统构成：一个**社交图**提供稳定的关系表征，三层**心理架构**提供实时行为决策。

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    NaNaGi 社交图 (Social Graph)                            │
│                    理论基础: Bowlby IWM [1][2][3] + GNN [19]              │
│                                                                          │
│                        南志锦 (admin)                                     │
│                        ╔═══════════╗                                     │
│                        ║ IWM Node  ║  "她心中的主人"                       │
│                        ║           ║                                     │
│                        ║ safety:   0.85   intimacy: 0.72                 │
│                        ║ care:     0.90   respect:  0.80                 │
│                        ║ reliable: 0.75   density:  0.90                 │
│                        ║ totalTurns: 156                                 │
│                        ╚════╤══════╝                                     │
│                             │ edge weight = intimacy × density           │
│                             │                                            │
│                       ┌─────┴─────┐                                      │
│                       │  NaNaGi   │                                      │
│                       │  (self)   │  ← 她的"性格硬件"，跨通道不变          │
│                       │           │     Schema Therapy [15]               │
│                       │ curiosity:  0.80   warmth:     0.75              │
│                       │ honesty:    0.90   autonomy:   0.70              │
│                       │ playfulness: 0.65   diligence: 0.85              │
│                       └─────┬─────┘                                      │
│                             │                                            │
│              ┌──────────────┼──────────────┐                             │
│              │              │              │                             │
│         克劳德 (uncle)   面试官A (guest-iv) 面试官B (guest-iv)             │
│         safety: 0.70    safety: 0.60     safety: 0.55                    │
│         intimacy:0.45   intimacy:0.15    intimacy:0.10                   │
│         density: 0.15   density: 0.05    density: 0.02                   │
│                                                                          │
│  ═══ 强连接 (density>0.5)    --- 弱连接    ··· Message Passing [4]       │
└──────────────────────────────────────────────────────────────────────────┘
```

### Self-Node — 她的性格硬件

理论基础：Young 图式疗法 [15] — 早期形成的核心人格结构稳定且难以改变。Self-Node 是 NaNaGi 的"出厂性格"，月~年极慢演化，跨通道完全不变。弹簧系数 K=0.05。

| trait | anchor | 含义 |
|-------|--------|------|
| curiosity | 0.80 | 对世界/他人的好奇心 |
| warmth | 0.75 | 天生的温暖度 |
| honesty | 0.90 | 诚实底线（不可撼动） |
| autonomy | 0.70 | 自主性需求强度 [14] |
| playfulness | 0.65 | 爱玩/爱闹的程度 |
| diligence | 0.85 | 认真程度 |

### IWM Nodes — 她心中的其他人

理论基础：Bowlby 内部工作模型 (Internal Working Model) [1][2][3] + Object Relations 内在客体 [16][17][18]。每个人在 NaNaGi 心中有一个独立的 IWM Node，6 个维度，弹簧拉回，随对话更新。**这不是"她对这个人的评价"，而是"她心中这个人的样子"——不等同于真实的对方。**

| trait | 含义 | 弹簧 K |
|-------|------|--------|
| safety | "这个人会不会伤害我？" [1] | adaptive |
| intimacy | "我们有多亲近？"（对话积累）[1] | adaptive |
| care | "我有多在意这个人？" | adaptive |
| respect | "这个人尊重我吗？" [14] | adaptive |
| reliability | "这个人说到做到吗？" | adaptive |
| understanding | "这个人理解我吗？" [13] | adaptive |

**弹簧系数来源**：Allostatic Load 理论 [10]——生物系统通过改变设定点适应长期压力。

```
K = max(0.10, 0.30 - density × 0.25)

density=0.05 → K=0.28  "刚认识"
density=0.30 → K=0.22  "开始熟悉"
density=0.60 → K=0.15  "关系稳定"
density=0.90 → K=0.08  "深厚信任，几乎不回退"
```

### 三层心理架构

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: 社交图 — 锚定网络 (月~年)                               │
│                                                                  │
│  Self-Node ←── edges ──→ IWM Nodes                              │
│  · 直接对话 → 更新对应 IWM Node                                   │
│  · 主人提到克劳德 → Heider 平衡 [4] → 克劳德节点更新               │
│    (仅 admin 通道触发; guest 提及任何人 → 不做图传播)               │
│  · 新用户 → 注册表单 → Cold Start 初始化 [19]                      │
│                                                                  │
│  引用: [1][2][3] Bowlby IWM / [16][17][18] Object Relations      │
│        [4] Heider 平衡 / [19] GraphSAGE                          │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2: 情绪空间 (分钟~小时)                                     │
│                                                                  │
│  六维情绪 (PAD [6] + Plutchik [7]):                               │
│    happiness / energy / dominance / intimacy / pride / calmness  │
│                                                                  │
│  · OCC 评价引擎 [8]（规则引擎，不经 LLM）:                          │
│    外部信号 → 目标相关性 × 期望一致性 × 因果归因 → EmotionDelta      │
│  · 双通路架构 [9]:                                                 │
│    低通路: 确定性规则 (<1ms, 每次必跑)                              │
│    高通路: LLM 内心独白 (条件触发, 1-3s)                            │
│  · 双弹簧拉回 [10]: Self K=0.05(极慢) / IWM K=adaptive            │
│  · 通道差异仅: 表达钳制范围 + 情绪弹簧松紧                          │
│                                                                  │
│  引用: [6] PAD / [7] Plutchik / [8] OCC / [9] LeDoux / [10]     │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 3: 社交规划 (秒~分钟)                                       │
│                                                                  │
│  Crick & Dodge SIP 六步决策 [11]:                                  │
│  编码线索 → 解释线索(感知IWM+ToM [13]) → 澄清目标 →                  │
│  生成策略(Gross 5策略池 [12]) → 评估选择 → 执行                     │
│                                                                  │
│  🔥 通道差异核心在 Step 3 (目标):                                   │
│    guest: 预设4目标（展示项目/了解兴趣/保持专业/引导展厅）            │
│    admin: 0义务，目标从对话涌现 [14]                                │
│                                                                  │
│  引用: [11] Crick&Dodge SIP / [12] Gross 情绪调节                 │
│        [13] Theory of Mind / [5] Jung Persona / [14] SDT         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三角色分层系统

NaNaGi 体系中有三种用户角色，外加 NaNaGi 自身作为系统角色。所有角色的 IWM Schema 和 Memory Schema 完全一致——唯一区别是存储位置和部分权限。

```
┌─────────────────────────────────────────────────────────────────┐
│  角色              权限                      存储位置             │
│                                                                  │
│  🦊 NaNaGi         —                       data/self/           │
│     (系统)          情绪/独白/自我模型        data/inner/         │
│                     不受任何用户直接修改                           │
│                                                                  │
│  👑 南志锦           全部权限                  data/admin/         │
│     (admin)         查看所有面试官反馈         文件系统            │
│                     查看内心独白              (可 cat/编辑)       │
│                     删除任何记忆                                   │
│                     查询 feedback 表                              │
│                     使用所有工具                                   │
│                     IWM 持久化到文件系统                            │
│                                                                  │
│  🎯 面试官           聊天 + 项目展厅            LevelDB            │
│     (guest-iv)      产生面试反馈 (feedback表)  iwm + mem +        │
│                     IWM 持久化                feedback + conv    │
│                     Cell 之间上下文隔离                            │
│                     可查看自己的记忆                               │
│                                                                  │
│  👤 普通用户         聊天 + 项目展厅            LevelDB            │
│     (guest)         IWM 持久化                iwm + mem + conv   │
│                     不产生反馈                                    │
│                     Cell 之间上下文隔离                            │
│                     (可配: 对话结束后是否丢弃数据)                   │
└─────────────────────────────────────────────────────────────────┘
```

### JWT 设计

```typescript
interface JWTPayload {
  personId: string;   // "nanzhijin" | "guest-V8k3mP2xQr6Z" | ...
  role: "admin" | "guest-iv" | "guest";
  name: string;       // "南志锦" | "张三" | ...
  identity: "主人" | "面试官" | "普通用户";
}
```

**personId 是节点的唯一标识**。从 GNN 的视角看，它就是 Node ID。所有 IWM 查找通过 personId 做 O(1) 直接索引，不做用户名扫描。

---

## 双通道分化

同一个 NaNaGi，同一个锚定人格 (Self-Node)，两种社交情境。理论基础：Jung 人格面具 [5] — 同一自我在不同社交场合呈现不同面向，但本质不变。

| | 面试官通道 (guest-iv) | 主人通道 (admin) |
|---|---|---|
| **Self-Node** | 完全相同 [15] | 完全相同 [15] |
| **IWM Node** | LevelDB 持久化，累积生长 | 文件系统持久化，累积生长 |
| **情绪表达** | 钳制 [0.3, 0.7] — Gross 反应调节 [12] | 不设限 [0.0, 1.0] — 真实表达 |
| **情绪弹簧** | K=0.3 (紧 — 快速回到职业稳定) | K=0.1 (松 — 情绪自然流动) |
| **社交目标** | 预设4个：展示项目/了解兴趣/保持专业/引导展厅 | 0义务，从对话涌现 [14] |
| **情境选择** | 不可用（不能不接待）[5] | 可用（拒绝权）[12][14] |
| **策略偏好** | 多用反应调节 [12]（藏情绪） | 少用反应调节 |
| **称呼** | 客人/您 | 主人 |
| **内心独白** | 不可见 | MemoryPanel 可查看 |
| **记忆读写** | LevelDB | 文件系统 + 隐形记忆 |
| **图消息传递** | 🚫 不触发（不与陌生人做图传播） | 主人提到第三者 → Heider 传播 [4] |
| **面试反馈** | ✅ 产生反馈 (feedback表) | — |
| **克劳德** | 不提 [5] | 叔叔，有独立 IWM Node |

### admin 通道的白板起点

admin 通道第一次对话也是白板起点（intimacy=0.1）。关系不是预设的，是从对话中生长出来的。与 Bowlby 的"安全基地从重复的敏感性照料中逐渐建立"一致 [1][2]。她对你的态度是她自己从经历里得出的结论，不是 System Prompt 预设的。

---

## 双图架构

NaNaGi 维护两张独立的图，各自有明确的用途和边界。

### Admin 通道：个人关系图 (Personal Graph)

```
南志锦 ←──→ 克劳德 ←──→ ...
  ↕
NaNaGi (Self)

用途: 维护 NaNaGi 对熟人网络的 IWM
触发: 主人提到某人 → Heider 平衡 [4] → 更新该人 IWM Node
影响: NaNaGi 对这个人的态度
```

### Guest 通道：面试反馈图 (Interview Feedback Graph)

```
面试官A ──[评价+岗位+项目问题]──→ 南志锦 (admin)
面试官B ──[评价+岗位+项目问题]──→ 南志锦 (admin)
面试官C ──[评价+岗位+项目问题]──→ 南志锦 (admin)

用途: 结构化记录面试官对南志锦的客观评价和需求
触发: 面试官表达评价/需求/问题 → NaNaGi 识别并记录
影响: NaNaGi 自己的 IWM 不受影响。主人事后可查询分析
存储: LevelDB feedback:{personId} — 每人一个 key，每次新对话追加记录
```

**两条图的核心区别**：个人关系图影响 NaNaGi 对熟人的态度（IWM 更新）；面试反馈图不影响 NaNaGi 自己的任何 IWM，纯粹是结构化数据收集，供主人查询。

### 查询流

```
主人: "最近一周有几个面试官来过？"
  → search_guests(days=7) → 扫描 LevelDB iwm:* + feedback:*
  → "最近7天有3位面试官: 张三(字节)来访2次、李四(腾讯)来访1次"

主人: "看看张三的反馈"
  → get_feedback(personId="guest-张三")
  → "张三，字节跳动Agent应用开发工程师。共来访2次。
     6月7日: 对GNN项目很感兴趣，问了冷启动和A/B测试。
     6月10日: 关注CnnMusic，认可不上双模态的决策。
     印象: 工程能力强，建议补充Agent框架内容。"
```

### feedback Schema (粗粒度)

```json
{
  "Key": "feedback:{guestPersonId}",
  "Value": {
    "personId": "guest-V8k3mP2xQr6Z",
    "name": "张三",
    "records": [
      {
        "timestamp": "2026-06-07T14:35:00Z",
        "sessionId": "sess-a1b2c3",
        "company": "字节跳动",
        "role": "Agent应用开发工程师",
        "impression": "对GNN项目评价高，问了很多工程落地细节，建议主人多准备Agent框架内容",
        "projectInterest": ["gnn", "cnn-music"],
        "quotes": ["你这个GNN项目跟我的方向很匹配", "冷启动具体怎么做的？"]
      }
    ],
    "summary": "字节跳动Agent方向面试官，共来访2次。关注GNN和CnnMusic。技术基础扎实。"
  }
}
```

---

## 环境感知：AmbientContext

每次对话开始前，从**时间·地点·天气**三件事推导 NaNaGi 的情绪基线。不经 LLM，确定性计算。理论基础：PAD 情绪模型的环境输入假说 [6] — 物理环境（光照、温度、空间）直接影响情绪三维度。

```
POST /api/chat
       │
       ▼
┌──────────────────────────────────────────┐
│  AmbientContext                           │
│                                           │
│  ⏰ 时间                                   │
│     timeOfDay: 黎明/早晨/上午/下午/        │
│                傍晚/夜晚/深夜 (7段) [6]     │
│     dayOfWeek: 工作日/周末                 │
│     season: 春夏秋冬                       │
│     isHoliday: 中国法定节假日              │
│     hoursSinceLastTalk: 上次对话距今 [1]   │
│     isFirstMeeting: 是否首次 [1]           │
│                                           │
│  📍 地点                                   │
│     request.ip → geoip-lite (MaxMind      │
│     GeoLite2, 本地数据库, 零网络延迟)       │
│     → city, country, timezone, coords     │
│     本地IP查不出 → null → 优雅降级          │
│                                           │
│  🌦 天气                                   │
│     coordinates → 和风天气 API             │
│     → condition, temperature, humidity,   │
│        windSpeed, visibility              │
│     sunlight: f(time, season, weather,    │
│                  latitude) 推导值 [6]      │
│     API 失败 → null → 优雅降级             │
│                                           │
│  → ambientMood (6维情绪基线偏移) [6][7]     │
│     happinessBias: 晴+0.05 / 雨-0.08     │
│     energyBias:   晨+0.08 / 深夜-0.08    │
│     calmnessBias: 风暴-0.15 / 晴+0.03   │
│     intimacyBias: 深夜+0.08 / 冬+0.04   │
│     dominanceBias: —                     │
│     prideBias:     —                     │
│                                           │
│  示例: 伦敦，下午3点，冬雨8°C，战时           │
│  → [0.31, 0.57, 0.10, 0.64, 0.48, 0.49]  │
│  (压抑但警觉，同舟共济)                      │
│  而非默认的 [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] │
└──────────────────────────────────────────┘
```

**AmbientContext 不包含 WHO(归 IWM 层)、WHAT/HOW(归 signals.ts)、WHY(归 planning.ts)。** 严格边界：只做时间·地点·天气 → 情绪基线偏移。

---

## 注册即 Cold Start

注册不是建账户，而是给 NaNaGi 的"第一印象"——这是 GNN Cold Start 的解决时刻。

### 引导式注册表单

```
┌─────────────────────────────────────────────────────────────────┐
│  你好～我是 NaNaGi ✨ 在开始之前，让我认识一下你吧                    │
│                                                                  │
│  📛 你的名字                                                     │
│  [_______________]                                               │
│                                                                  │
│  🎭 你的身份                                                     │
│  ○ 我是来面试南志锦的 (面试官)                                     │
│  ○ 我只是随便看看 (普通用户)                                       │
│                                                                  │
│  ── 面试官专属 ↓ ──────────────────────────────────────         │
│                                                                  │
│  🏢 公司                                                        │
│  [_______________]  例: 字节跳动                                  │
│                                                                  │
│  💼 招聘岗位                                                     │
│  [_______________]  例: Agent应用开发工程师                        │
│                                                                  │
│  🔍 我关注的技术方向 (可多选)                                      │
│  □ 推荐系统    □ GNN/图神经网络  □ Agent/AI Agent                  │
│  □ 计算机视觉  □ NLP           □ 音频/音乐检索                     │
│  □ 后端工程    □ 数据处理/Spark □ 模型部署/MLOps                    │
│  □ 其他: [_______________]                                       │
│                                                                  │
│  📋 我想了解 (可多选)                                              │
│  □ 南志锦的GNN社交图谱项目                                        │
│  □ CnnMusic多模态召回项目                                         │
│  □ FruitCNN水果识别项目                                           │
│  □ 南志锦的整体技术能力                                           │
│  □ 项目背后的设计决策                                             │
│                                                                  │
│  ── 通用 ↓ ──────────────────────────────────────────────      │
│                                                                  │
│  🔑 设置密码 (下次用这个登录)                                      │
│  [_______________]                                               │
│                                                                  │
│  [开始对话 ✨]                                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 后端处理

表单直接映射到 IWM 初始化，不需要 NLP 提取：

```typescript
function initIWMFromForm(form: RegisterForm): IWMNode {
  return {
    personId: `guest-${nanoid(12)}`,
    name: form.name,
    role: form.identity === "面试官" ? "guest-iv" : "guest",
    identity: form.identity,
    
    traits: IDENTITY_BASELINE[form.identity],  // 身份 → 初始 traits
    
    knownFacts: [
      form.identity === "面试官"
        ? [`${form.company}面试官`, `招聘${form.role}`]
        : [],
      ...form.techInterests.map(t => `关注${t}`),
      ...form.wantToKnow.map(w => `想了解${w}`),
    ].flat().filter(Boolean),
    
    topicInterests: mapToProjects(form.techInterests, form.wantToKnow),
    company: form.company || null,
    jobRole: form.role || null,
    
    firstMet: new Date().toISOString(),
    totalTurns: 0,
    historyDensity: 0.0,
  };
}
```

### 登录

```
personId + password → LevelDB O(1) get('user:{personId}')
→ bcrypt.compare → 签发 JWT { personId, role, name, identity }
→ 加载已有 IWM Node
```

**personId 即节点唯一标识。不存在"用户名搜索"。O(1) 直接索引。**

---

## Cell 隔离记忆架构

与 ChatGPT、Claude 等主流大模型产品同构设计：每个对话 Cell 是一个独立的上下文容器，**存储该 Cell 内的完整聊天历史（每条消息都持久化）**。用户打开某个 Cell 时，历史全量加载到前端，可以接着上次继续聊。不同 Cell 之间的聊天历史互相隔离，不进彼此的 LLM 上下文。

```
┌──────────────────────────────────────────────────────────────────┐
│  Cell 1 (GNN面试)           Cell 2 (随便聊聊)        Cell 3 (空)  │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────┐   │
│  │ 消息1: 你好...     │    │ 消息1: 在吗...     │    │          │   │
│  │ 消息2: 我是字节...  │    │ 消息2: 嗯嗯       │    │          │   │
│  │ ...               │    │ ...               │    │          │   │
│  │ 消息18: 谢谢！     │    │ 消息5: 拜拜        │    │          │   │
│  │                   │    │                   │    │          │   │
│  │ 全部持久化在数据库  │    │ 全部持久化在数据库  │    │ 新建即空   │   │
│  │ 打开 = 全部可见     │    │ 打开 = 全部可见     │    │          │   │
│  │ 可继续聊,追加消息   │    │ 可继续聊,追加消息   │    │          │   │
│  └──────────────────┘    └──────────────────┘    └──────────┘   │
│         │                       │                      │        │
│         └───────────────────────┴──────────────────────┘        │
│                                 │                                │
│                   跨 cell 累积 (不进 LLM 上下文)                   │
│                                 │                                │
│         ┌───────────────────────┼────────────────────┐          │
│         ▼                       ▼                    ▼          │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ IWM Node    │    │ 显性记忆      │    │ 面试反馈      │       │
│  │ (6 traits)  │    │ (mem表)      │    │ (feedback表)  │       │
│  │ 跨cell累积   │    │ 跨cell累积    │    │ 仅面试官      │       │
│  └─────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
│  每个 cell 启动时注入 System Prompt (不超 ~1500 tokens):          │
│    [关系感知] ← IWM 摘要 (~150字)                                 │
│    [已有记忆] ← 显性记忆 (已有机制)                                │
│    [上次对话] ← 上一 cell 摘要 (~200字, conv 表)                  │
│                                                                  │
│  ❌ 不注入: 当前 cell 的完整聊天历史之外的任何内容                   │
│  ✅ 当前 cell 内的所有消息作为 messages[] 传入 LLM                  │
└──────────────────────────────────────────────────────────────────┘
```

### Cell 数据模型

```typescript
// Cell 的完整数据结构
interface CellRecord {
  cellId: string;                    // "cell-{nanoid}"
  personId: string;                  // 所属用户
  title: string;                     // 前端显示标题 ("GNN项目面试")
  createdAt: string;                 // 创建时间
  lastMessageAt: string;             // 最后一条消息时间
  messageCount: number;              // 消息总数
  isActive: boolean;                 // 是否仍在对话中
  
  // 🔑 核心: 该 Cell 内的所有聊天消息, 每条都持久化
  messages: Message[];               // [{id, role, content, timestamp}, ...]
  
  // 可选元数据
  summary?: string;                  // Cell 结束时生成的摘要 (用于跨 Cell 注入)
  toolCallsUsed?: string[];          // 本 Cell 调用了哪些工具
}
```

### 用户交互流

```
1. 用户登录 → 前端加载 Cell 列表 (从 conv 表扫描该用户的所有 Cell)
   左侧显示: [GNN项目面试 (6/7, 18条)] [随便聊聊 (6/10, 5条)] [+ 新对话]

2. 用户点击某个 Cell → 前端加载该 Cell 的 messages[] 全量
   → 聊天界面显示完整历史
   → 用户输入新消息 → 追加到 messages[] → 持久化到数据库
   → 可以无限继续聊下去

3. 用户点击 [+ 新对话] → 创建新 Cell
   → messages = [] (空白)
   → System Prompt 注入 IWM 摘要 + 上次 Cell 摘要
   → 开始全新对话

4. 用户切换到另一个 Cell → 当前 Cell 的消息持久化
   → 加载另一个 Cell 的 messages[]
   → 两个 Cell 的上下文完全隔离
```

### Cell 间上下文隔离

**打开 Cell 1 (GNN面试)**:
```
LLM 收到的 messages:
  [system] ← 七段式 Prompt (含 IWM 摘要 + 上轮摘要, ~1500 tokens)
  [user]   你好，我是字节的面试官...
  [assistant] 张三您好～欢迎光临 ✨ ...
  [user]   能介绍一下你主人的 GNN 项目吗？
  [assistant] 当然！主人的 GNN 项目是 CAAI-BDSC2023...
  ... (该 Cell 内全部 18 条消息)
  
  Cell 2 的 5 条消息 → ❌ 不在此上下文中
```

**用户切换到 Cell 2 (随便聊聊)**:
```
LLM 收到的 messages:
  [system] ← 七段式 Prompt (相同 IWM 摘要, 不同上轮摘要)
  [user]   在吗
  [assistant] 在的呢～主人不在，但我可以陪你聊 ✨
  ... (该 Cell 内全部 5 条消息)
  
  Cell 1 的 18 条消息 → ❌ 不在此上下文中
```

### 为什么不用全量历史进上下文

1. **上下文膨胀**: LLM 注意力机制在 >8K tokens 后推理质量下降
2. **Token 成本**: 每个请求都带全部历史 → 成本线性增长
3. **信息污染**: 旧对话中的错误/误导信息会干扰新对话
4. **主流实践**: ChatGPT、Claude 都是 Cell 隔离——他们这样做是有原因的

### Cell 摘要 (跨 Cell 上下文桥梁)

每个 Cell 结束时，用独立轻量 LLM 调用生成 ~200 字摘要。新 Cell 启动时，注入上一个 Cell 的摘要（仅一个），让 NaNaGi 在新对话中对"上次聊了什么"有基本认知。

```typescript
async function summarizeCell(messages: Message[]): Promise<string> {
  const response = await fetch(DEEPSEEK_URL, {
    body: JSON.stringify({
      model: "deepseek-chat",
      max_tokens: 300,
      messages: [
        { role: "system", content: "将对话总结为200字以内的摘要。包含: 对方是谁、聊了什么、值得下次参考的信息。" },
        { role: "user", content: messages.map(m => `${m.role}: ${m.content}`).join('\n') }
      ]
    })
  });
  return response.json().choices[0].message.content;
}
```

### 前端 Cell 列表

```
┌──────────────────────────────────────────────────────┐
│  NaNaGi                                     [+新对话] │
├──────────┬───────────────────────────────────────────┤
│          │                                           │
│ 📋 历史   │         Cell 内容区                        │
│          │                                           │
│ GNN面试   │  张三: 你好，我是字节的面试官...              │
│ 6/07     │  NaNaGi: 张三您好～欢迎光临 ✨               │
│ 18条消息  │  听说您是字节的面试官，做Agent方向的？          │
│          │  ...                                      │
│ 随便聊聊  │                                           │
│ 6/10     │  ┌─────────────────────────────┐          │
│ 5条消息   │  │ 输入消息...                   │          │
│          │  └─────────────────────────────┘          │
│ + 新对话  │                                           │
│          │                                           │
└──────────┴───────────────────────────────────────────┘
```
- 左侧 Cell 列表，按最后对话时间倒序
- 点击 Cell → 加载该 Cell 完整 messages[] → 显示全部历史
- 新建 Cell → 空白对话 + IWM 摘要注入
- 不同 Cell 上下文完全隔离

---

## 完整对话数据流

```
POST /api/chat
  → JWT { personId, role, name, identity }
      │
      ▼
┌─ STEP 0: 加载 IWM Node ──────────────────────────┐
│  role=admin → data/admin/nanzhijin-iwm.json      │
│  role=guest* → LevelDB get('iwm:{personId}')     │
│  新节点 → 从注册表单数据初始化                      │
│  根据 elapsed time 计算弹簧拉回 [10]               │
└────────────────────┬──────────────────────────────┘
                     ▼
┌─ STEP 1: 环境感知 [6][7] ─────────────────────────┐
│  request.ip → geoip-lite → city/coords            │
│  coords → 和风API → 天气                           │
│  时间 → timeOfDay(7段)/season/holiday/对话间隔     │
│  → ambientMood (6维情绪基线偏移)                    │
└────────────────────┬──────────────────────────────┘
                     ▼
┌─ STEP 2: 外部信号提取 (signals) ──────────────────┐
│  情感词典扫描 / 句法模板 / 消息元数据 / 提及检测      │
│  → ExternalSignals (确定性算法，不经 LLM)           │
└────────────────────┬──────────────────────────────┘
                     ▼
┌─ STEP 3: OCC 情绪评价 [8] ────────────────────────┐
│  signals → OCC 3维评价 → EmotionDelta              │
│  感知 IWM Node [1]: respect高 → 批评被解释为帮助    │
│  更新情绪 + 更新当前 IWM Node                       │
└────────────────────┬──────────────────────────────┘
                     ▼
┌─ STEP 4: 双弹簧拉回 [10] + 通道钳制 ──────────────┐
│  Self K=0.05 (极慢) / IWM K=adaptive              │
│  guest clamp[0.3, 0.7] / admin clamp[0, 1]        │
│  钳制对应 Gross 反应调节策略 [12]                    │
└────────────────────┬──────────────────────────────┘
                     ▼
┌─ STEP 4.5: 图消息传递 ────────────────────────────┐
│  [条件: role=admin + mentionsPerson]                │
│  主人提到克劳德 → Heider 平衡传播 [4] → 节点更新     │
│  guest 通道: 不触发 (不做图传播)                     │
└────────────────────┬──────────────────────────────┘
                     ▼
┌─ STEP 5: 内心独白 [条件触发] ──────────────────────┐
│  触发: |Δ|>0.15 OR selfDisclosure OR round%5==0   │
│        OR mentionsPerson OR firstMeeting           │
│  高通路 LLM 调用 [9] (max_tokens=200, 无 tools)    │
│  → ReflectionText → data/self/inner/               │
└────────────────────┬──────────────────────────────┘
                     ▼
┌─ STEP 6: 社交规划 SIP [11] ───────────────────────┐
│  编码→解释(感知IWM+ToM [13])→澄清目标→              │
│  生成策略(Gross 5策略池 [12])→评估选择→执行           │
│  guest-iv/guest: 预设4目标                          │
│  admin: 0义务，目标从对话涌现 [14]                    │
│  → SocialPlan                                      │
└────────────────────┬──────────────────────────────┘
                     ▼
┌─ STEP 7: 人格过滤 ────────────────────────────────┐
│  Emotion + IWM + Plan → PersonaParameters [5]      │
│  8维语气参数 (warmth, formality, playfulness...)   │
│  IWM感知: intimacy高→warmth高, safety高→playful    │
│  确定性映射，不经 LLM                                │
└────────────────────┬──────────────────────────────┘
                     ▼
┌─ STEP 8: System Prompt 组装 ──────────────────────┐
│  [0] 环境感知 (ambient mood) [6]                   │
│  [1] 角色层 (role + identity → 身份描述) [5]        │
│  [2] 关系感知 (IWM 摘要: "你们认识了X次...") [1]    │
│  [3] 人格注入 (filter 输出 → 语气参数)              │
│  [4] 上次对话 (~200字, 从 conv 表取最近 cell 摘要)   │
│  [5] 已有记忆 (显性记忆, 现有机制)                   │
│  [6] 工具层 (tool descriptions)                    │
│  [7] 行为准则                                       │
│                                                    │
│  总长度: ~1500 tokens → 不超模型上下文               │
└────────────────────┬──────────────────────────────┘
                     ▼
┌─ STEP 9: ReAct 循环 ──────────────────────────────┐
│  while round < 5:                                  │
│    LLM(systemPrompt + messages + tools)            │
│    text → SSE 推送 → break                          │
│    tool_use → 循环检测(hash) → 执行 → 注入 → 继续   │
│                                                    │
│  三层容灾: 30s超时 → 重试1次 → 降级回复              │
│    降级措辞因 role 不同:                             │
│      guest: "抱歉呢，服务暂时有点小麻烦～"            │
│      admin: "主人...大脑好像卡住了，能等一下吗？"      │
└────────────────────┬──────────────────────────────┘
                     ▼
┌─ STEP 10: 后处理 ─────────────────────────────────┐
│  情绪持久化 → data/self/emotion-state.json         │
│  IWM 持久化 → admin:fs / guest:LevelDB             │
│  情绪审计 → data/self/emotion-log.jsonl (全局)     │
│  对人情绪轨迹 → iwm:{personId} (LevelDB)            │
│  内心独白 → data/self/inner/inner-{ts}.md          │
│  显性记忆 → admin:fs / guest:LevelDB               │
│  Cell 摘要 → LevelDB conv:{personId}:{cellId}       │
│  面试反馈 → LevelDB feedback:{personId}             │
│    (仅 guest-iv, 对话中检测到评价/岗位信号时)        │
└───────────────────────────────────────────────────┘
```

---

## 三层分级存储架构

### 设计原则

```
┌─────────────────────────────────────────────────────────────────┐
│  不是所有数据都应该进同一个存储系统。                                  │
│                                                                  │
│  NaNaGi 本体的数据 (情绪、独白、自我模型):                           │
│    → 文件系统。每次对话必读, sub-ms 延迟。可 cat/编辑器直接看。       │
│                                                                  │
│  南志锦的数据 (认证、IWM、记忆):                                    │
│    → 文件系统。与 NaNaGi 本体同层但独立目录。可审计、可编辑。          │
│                                                                  │
│  所有 guest 用户的数据 (认证、IWM、记忆、反馈、Cell):                │
│    → LevelDB。O(1) key 查找。嵌入式，零网络。事务安全。              │
│                                                                  │
│  显性记忆 (跨角色):                                                │
│    → admin 用文件系统 (可审计哲学保留)                               │
│    → guest 用 LevelDB (扩展性 + 数据安全)                           │
└─────────────────────────────────────────────────────────────────┘
```

### 目录结构

```
data/
├── self/                          ← 🦊 NaNaGi 本体 (文件系统, sub-ms)
│   ├── self-node.json             ← Self-Node: 7 traits + anchor
│   ├── emotion-state.json         ← 当前 6维情绪 + ambientMood
│   ├── emotion-log.jsonl          ← 情绪变化审计 (append-only)
│   └── inner/                     ← 内心世界
│       └── inner-{ts}.md          ← 内心独白文本
│
├── admin/                         ← 👑 南志锦专属 (文件系统, 可审计)
│   ├── nanzhijin.json             ← 主人认证 {passwordHash, role}
│   ├── nanzhijin-iwm.json         ← 主人 IWM Node
│   └── memories/                  ← 主人相关显性记忆
│       ├── MEMORY.md              ← 记忆索引
│       └── mem-{ts}.md            ← 记忆文件 (YAML frontmatter)
│
├── leveldb/                       ← 🌐 所有 guest 用户
│   │                                classic-level (MIT许可证)
│   │                                纯本地 C++ 编译, 零网络依赖
│   │                                不依赖任何 Google 服务器
│   │
│   ├── user:{personId}            ← 用户认证表
│   ├── iwm:{personId}             ← IWM 节点表
│   ├── mem:{personId}:{ts}        ← 用户记忆表
│   ├── emo:{personId}:{ts}        ← 情绪轨迹表
│   ├── conv:{personId}:{cellId}   ← Cell 会话表
│   └── feedback:{personId}        ← 🆕 面试反馈表 (仅 guest-iv)
│
└── memory/                        ← 显性记忆 (现有, V2.5)
    ├── MEMORY.md
    └── mem-{ts}.md
```

### 统一接口

```typescript
// lib/store.ts — 统一数据访问层
// 调用方不感知底层是文件系统还是 LevelDB

// IWM
store.getNode(personId) → IWMNode | null
store.putNode(personId, node) → void

// Memory
store.createMemory(record) → void
store.listMemories(personId) → MemoryEntry[]

// Cell
store.createCell(personId, cellId, messages) → void
store.getCellSummary(personId, cellId) → string

// Feedback (仅 guest-iv)
store.appendFeedback(personId, record) → void
store.getFeedback(personId) → FeedbackRecord | null
```

### 为什么是 LevelDB

| 方案 | 适合？ | 理由 |
|------|--------|------|
| 文件系统 | ❌ (用户层) | 并发写不安全，无事务，5000+用户后目录膨胀 |
| SQLite | 🟡 | 功能满足，但设计红线里文件系统哲学保留给 NaNaGi 本体 |
| **LevelDB** | ✅ | 嵌入式 K-V，MIT 许可证，Chrome 内置同款引擎，百万级 O(1)，零网络依赖 |
| LanceDB | 🟡 | 计划做向量检索 (P2)。K-V 不是它最佳场景 |
| Redis | ❌ | 需独立部署，对个人项目过重 |
| PostgreSQL | ❌ | 同上 |
| Spark | ❌ | 离线批处理引擎，登录延迟 30 秒以上 |

**JD 对齐**：职责 3 (性能优化) → LevelDB O(1) + 嵌入式无网络开销；职责 4 (框架沉淀) → `lib/store.ts` 接口可替换；职责 6 (上线落地) → 文件系统不安全 → LevelDB 事务安全。

---

## LevelDB 六表 Schema

### Table 1: `user` — 用户认证

```
Key:   "user:{personId}"
Value: {
  personId, name, passwordHash (bcrypt),
  role ("guest-iv" | "guest"),
  identity ("面试官" | "普通用户"),
  company, jobRole (面试官专属),
  techInterests[], wantToKnow[],
  createdAt, lastLogin
}
```

### Table 2: `iwm` — IWM 节点

```
Key:   "iwm:{personId}"
Value: {
  personId, name, role, identity,
  traits: { safety, intimacy, care, respect, reliability, understanding },
  anchor: { safety: {value, springK}, ... },
  knownFacts[], topicInterests[],
  company, jobRole,  // 面试官专属
  firstMet, lastTalk, totalTurns, historyDensity,
  emotionTimeline: [{date, avgHappiness, avgIntimacy}]
}
```

### Table 3: `mem` — 用户记忆

```
Key:   "mem:{personId}:{timestamp}"
Value: {
  slug, personId,
  meta: { name, description, type, tags, createdAt },
  content (YAML frontmatter + Markdown),
  summary, keywords[]
}
```

### Table 4: `emo` — 情绪轨迹

```
Key:   "emo:{personId}:{timestamp}"
Value: {
  timestamp, personId, roundNumber,
  before: { happiness, energy, dominance, intimacy, pride, calmness },
  delta: { happiness, energy, dominance, intimacy, pride, calmness },
  trigger: { type ("praise"|"criticism"|"selfDisclosure"|...), summary },
  ambientMood: { happinessBias, energyBias, calmnessBias, ... }
}
```

### Table 5: `conv` — Cell 会话 (完整聊天历史)

```
Key:   "conv:{personId}:{cellId}"
Value: {
  cellId:       "cell-a1b2c3",
  personId:     "guest-V8k3mP2xQr6Z",
  title:        "GNN项目面试",            // 前端显示标题
  createdAt:    "2026-06-07T14:30:00Z",
  lastMessageAt:"2026-06-07T14:52:00Z",
  messageCount: 18,
  
  // 🔑 核心: 该 Cell 内的所有聊天消息, 每条都持久化
  messages: [
    { id: "msg-1", role: "user",      content: "你好, 我是字节的面试官...", timestamp: "..." },
    { id: "msg-2", role: "assistant", content: "张三您好～欢迎光临 ✨...",  timestamp: "..." },
    { id: "msg-3", role: "user",      content: "能介绍一下GNN项目吗?",     timestamp: "..." },
    // ... 全部 18 条消息
  ],
  
  // 元数据
  toolCallsUsed: ["get_project_info", "search_web"],
  emotionStart:  { happiness: 0.60, ... },  // Cell 开始时的情绪
  emotionEnd:    { happiness: 0.68, ... },  // Cell 结束时的情绪
  summary:       "字节面试官张三, 22分钟, 主要讨论GNN冷启动和A/B测试"  // Cell结束后自动生成
}

额外 Key:
  "conv:{personId}:cells" → string[] (该用户的所有 cellId 列表, 用于前端 Cell 列表)
  "conv:{personId}:last-summary" → string (最新 cell 摘要, 新 cell 注入用)
```

### Table 6: `feedback` — 面试反馈 (仅 guest-iv)

```
Key:   "feedback:{personId}"
Value: {
  personId, name,
  records: [{
    timestamp, sessionId, cellId,
    company, role, impression (自然语言),
    projectInterest[], quotes[]
  }],
  summary (NaNaGi 自动更新的总结)
}
```

---

## 存储架构：设计 vs 当前实现 (P3 阶段)

> 以下对照说明"为什么这样实现"以及"什么阶段会补全"，面试时可展示工程决策过程。

### 引擎选型

| | 设计目标 | P3 实际实现 | 原因 |
|------|---------|------------|------|
| guest 存储引擎 | classic-level (真 LevelDB) | **文件系统 K-V + 用户子目录** | Turbopack 不兼容 native C++ 模块 (geoip-lite、classic-level 均报错) |
| 部署后 | — | 换真 LevelDB | `lib/leveldb.ts` 接口封装完整，换引擎只改此文件，签名不变 |
| admin 存储 | 文件系统 | ✅ 文件系统 | 设计哲学保留——可 cat/编辑器直接看 |

### 目录结构

| 设计 | P3 实际 | 差异原因 |
|------|---------|---------|
| 扁平 key: `user:{personId}` | 用户子目录: `{personId}/user.json` | Windows 文件名不允许冒号 `:` |
| `data/self/` | ❌ 未创建 | P5 人格引擎层依赖 (Self-Node、情绪状态、内心独白) |
| `data/leveldb/` 六表全建 | 部分: user/iwm/mem ✅, emo/conv/feedback 📋 | emo→P5情绪引擎, conv→P4 Cell, feedback→P6面试反馈 |
| `data/admin/nanzhijin.json` | 🟡 认证走 `lib/auth.ts` bcrypt，文件未创建 | P5 补 |

### IWM Node 结构

| 字段 | 状态 |
|------|------|
| traits (6维) | ✅ |
| knownFacts, topicInterests | ✅ |
| totalTurns, lastTalk, historyDensity | ✅ |
| anchor (每维度 springK) | 📋 P5-3 情绪引擎实现后补 |
| emotionTimeline | 📋 P5-3 |

### Key 格式对照

```
设计:                                   P3 实际:
data/leveldb/                           data/leveldb/
  user:guest-xxx.json                     guest-xxx/
  iwm:guest-xxx.json                        user.json
  mem:guest-xxx:ts.json                     iwm.json
                                            memories/{ts}.json
                                          _index/email/{key}.json
```

**面试话术**："LevelDB 是目标引擎，但 Turbopack 开发环境不兼容 C++ native 模块。我用文件系统 K-V 替代——接口完全一致，部署时换真 LevelDB 只改一个文件。Windows 文件名不支持冒号，所以扁平 key 改成了用户子目录——反而更干净，每个用户数据物理隔离。"

---

## GNN 概念映射

NaNaGi 的社交图与南志锦的 GNN 社交图谱链接预测项目形成学术对称 [19]：

| GNN 概念 | NaNaGi 对应 | 心理学对应 |
|---------|------------|-----------|
| Node Embedding | Self/IWM node traits 向量 | Internal Working Model [1] |
| Edge Weight | intimacy × density | 依恋强度 [1] |
| Message Passing | 主人提到克劳德 → 沿边传播 (仅 admin) | Heider 平衡 [4] |
| Link Prediction | 新 guest 值得信任多少？ | 社会认知 [13] |
| Cold Start | 注册表单 → 初始化 IWM Node | 第一印象形成 [1] |
| GraphSAGE Aggregation [19] | 聚合所有已知节点 → 全局状态 | Object Relations [16][17] |
| Embedding Update | 每轮对话更新 IWM 维度 | 印象更新 |

---

## 待研究：IWM 更新触发条件

> 当前 P2 实现为每次对话后自动 totalTurns++/lastTalk 更新。
> 真实的 IWM 变化应仅在情感显著性事件触发——日常闲聊不动锚点。
> 参考 Bowlby + Bretherton 的 IWM 稳定性研究，触发条件与信号强度阈值需要真实对话数据校准。
> 此问题将在上线后通过 audit log 分析确定，作为 P5-B 或后续迭代的输入。

### 待确定项

| 问题 | 当前状态 | 需要什么 |
|------|---------|---------|
| 信号强度阈值 | 未定义 (praise=?/criticism=?/selfDisclosure=?) | 上线后从 emotion-log.jsonl 分析分布 |
| IWM springK 精确值 | K=0.02~0.05 估计 | 多用户长期对话数据 |
| 触发分类准确性 | signals.ts 用正则+词典 | 真实对话样本测试 |

---

## 学术支撑

本项目参考了 19 个心理学和机器学习理论模型。每个引用对应一个具体的设计模块。

### 参考文献

1. **Bowlby, J.** (1969). *Attachment and Loss, Vol. 1: Attachment*. New York: Basic Books. — 依恋理论与内部工作模型 (IWM) 的基础。

2. **Bowlby, J.** (1973). *Attachment and Loss, Vol. 2: Separation: Anxiety and Anger*. New York: Basic Books. — 安全基地的形成与破坏。IWM Node safety 维度 + 弹簧拉回动力学来源。

3. **Bretherton, I., & Munholland, K. A.** (2008). Internal working models in attachment relationships: Elaborating a central construct in attachment theory. In J. Cassidy & P. R. Shaver (Eds.), *Handbook of Attachment* (pp. 102–127). Guilford Press. — IWM 作为图节点的理论支持。

4. **Heider, F.** (1958). *The Psychology of Interpersonal Relations*. New York: Wiley. — P–O–X 平衡理论。社交图 Message Passing 机制来源 (仅 admin 通道)。

5. **Jung, C. G.** (1953). The persona as a segment of the collective psyche. In *Collected Works, Vol. 7: Two Essays on Analytical Psychology* (R. F. C. Hull, Trans., pp. 156–171). Princeton University Press. (Original work published 1943) — 人格面具：同一自我在不同社交情境中呈现不同面向。双通道系统的理论基础。

6. **Mehrabian, A., & Russell, J. A.** (1974). *An Approach to Environmental Psychology*. Cambridge, MA: MIT Press. — PAD 情绪三维模型 (Pleasure–Arousal–Dominance)。物理环境对情绪的影响 — AmbientContext 的环境输入假说。

7. **Plutchik, R.** (1980). *Emotion: A Psychoevolutionary Synthesis*. New York: Harper & Row. — 情绪轮理论。intimacy/pride/calmness 三个额外维度的选择依据。

8. **Ortony, A., Clore, G. L., & Collins, A.** (1988). *The Cognitive Structure of Emotions*. Cambridge University Press. — OCC 评价模型：情绪源于对事件的三维认知评价 (目标相关性/期望一致性/因果归因)。Step 3 OCC 情绪评价引擎的直接映射。

9. **LeDoux, J. E.** (1996). *The Emotional Brain: The Mysterious Underpinnings of Emotional Life*. New York: Simon & Schuster. — 双通路情绪理论：低通路 (杏仁核快路, <20ms) 与高通路 (皮层慢路, 300–500ms)。低通路 = OCC 规则引擎；高通路 = 内心独白 LLM 调用。

10. **McEwen, B. S., & Stellar, E.** (1993). Stress and the individual: Mechanisms leading to disease. *Archives of Internal Medicine*, 153(18), 2093–2101. — Allostatic Load 理论：生物系统通过改变设定点 (allostasis) 适应长期压力。弹簧力学 (K 系数 + 锚点拉回) 的生物学基础。

11. **Crick, N. R., & Dodge, K. A.** (1994). A review and reformulation of social information-processing mechanisms in children's social adjustment. *Psychological Bulletin*, 115(1), 74–101. — SIP 六步社交信息加工模型。Step 6 社交规划的直接蓝图。

12. **Gross, J. J.** (1998). The emerging field of emotion regulation: An integrative review. *Review of General Psychology*, 2(3), 271–299. — 五大情绪调节策略：情境选择/情境修正/注意分配/认知重评/反应调节。SIP Step 4 策略池 + 通道差异的来源。

13. **Premack, D., & Woodruff, G.** (1978). Does the chimpanzee have a theory of mind? *Behavioral and Brain Sciences*, 1(4), 515–526. — Theory of Mind：推断他人心理状态的能力。SIP Step 2 (解释线索) + IWM Node understanding 维度的认知基础。

14. **Deci, E. L., & Ryan, R. M.** (2000). The "what" and "why" of goal pursuits: Human needs and the self-determination of behavior. *Psychological Inquiry*, 11(4), 227–268. — 自我决定理论：自主性/胜任感/关联性三大基本心理需求。admin 通道 0 义务设计 (自主性) + IWM Node 的关系维度 (关联性)。

15. **Young, J. E., Klosko, J. S., & Weishaar, M. E.** (2003). *Schema Therapy: A Practitioner's Guide*. New York: Guilford Press. — 图式疗法：早期形成的核心人格结构稳定且难以改变。Self-Node 作为"性格硬件" (K=0.05 极慢演化)。

16. **Klein, M.** (1946). Notes on some schizoid mechanisms. *International Journal of Psycho-Analysis*, 27, 99–110. — 内在客体与投射性认同的原初论证。IWM Node 作为 representation-not-reality 的核心理论锚点。

17. **Klein, M.** (1957). *Envy and Gratitude*. London: Tavistock. — 内在客体世界的情感动力：嫉羡与感恩如何塑造对他人表征的态度。IWM Node 的 care 与 respect 维度来源。

18. **Winnicott, D. W.** (1965). *The Maturational Processes and the Facilitating Environment*. London: Hogarth Press. — 促进性环境：健康的心理发展需要足够好的照料。IWM Node 的 safety 与 intimacy 维度来源。

19. **Hamilton, W. L., Ying, R., & Leskovec, J.** (2017). Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems*, 30. — GraphSAGE：归纳式图节点嵌入学习。社交图的 Message Passing 与节点更新机制的数学对应。

---

## 当前状态与路线图

### 已实现 ✅ (V2.5 + P1)

| 功能 | 说明 | 版本 |
|------|------|------|
| NaNaGi Agent 对话 | DeepSeek V4 Pro 引擎，Anthropic 兼容端点，流式 SSE | V1 |
| 密码鉴权 | bcrypt 双密码 + JWT + httpOnly cookie | V1 |
| 混元生图 | 腾讯混元 hy-image-v3.0，异步 submit + 轮询 query | V2 |
| 可拖拽图片 | 聊天框内图片可拖拽，松手弹回，带下载按钮 | V2 |
| 唱片机互动 | 项目页拖图片进唱片机 → 魔法扫描识别动画 | V2 |
| 记忆系统 | 文件记忆双路径架构 | V2.5 |
| 记忆面板 | 左侧滑出，像素风卡片，悬浮放大，管理员删除 | V2.5 |
| 记忆注入 | 每次对话自动注入已有记忆到 System Prompt | V2.5 |
| 聊天持久化 | sessionStorage 跨页面导航保持 + 刷新恢复 | V2 |
| 项目展厅 | 3 个项目页 (FruitCNN / CnnMusic / GNN)，SSG 预渲染 | V1 |
| 三风格设计系统 | 女仆围裙 + 像素下午茶 + 星尘备忘录 | V2 |
| **ReAct 循环** | 5 轮 + hash 循环检测 + 三层容灾 (30s超时→重试→降级) | 🆕 P1 |
| **工具注册表** | 8 个工具，Map<name, ToolEntry>，并行执行 | 🆕 P1 |
| **双通道 Prompt** | guest(女仆)/admin(主人) 七段式动态拼接 | 🆕 P1 |
| **环境感知** | 时间(7段+季节+节假日) + IP定位(ip-api.com) + 天气(和风API,1h缓存) | 🆕 P1 |
| **安全配置中心** | lib/env.ts 集中管控所有 API Key，import "server-only" 防泄漏 | 🆕 P1 |
| **Agent 架构** | route.ts 644→160行，agent/personality 模块化 | 🆕 P1 |
| **统一数据访问层** | lib/store.ts 统一接口: admin→文件系统, guest→文件K-V | 🆕 P2 |
| **IWM 节点存储** | 六表 K-V (user/iwm/mem/emo/conv/feedback)，文件系统实现 | 🆕 P2 |
| **IWM 自动持久化** | 每次对话结束自动更新 IWM Node (totalTurns/lastTalk/density) | 🆕 P2 |
| **邮箱验证码注册** | QQ邮箱 SMTP + 6位验证码 + 格式验证 + 一次性邮箱拦截 + 拼写纠错 | 🆕 P3 |
| **邮箱登录** | email→personId 索引 + bcrypt验证 + JWT(personId/role/name/identity) | 🆕 P3 |
| **用户数据隔离** | 每人独立子目录: {personId}/user.json + iwm.json + memories/ | 🆕 P3 |
| **记忆按角色过滤** | System Prompt注入 + MemoryPanel API + 工具路由 均按personId隔离 | 🆕 P3 |
| **环境感知节制** | 24h全查询缓存: 时间每次<1ms / 地点+天气每天只查一次 | 🆕 P3 |
| **Cell 对话系统** | 对话列表+滑出面板+消息持久化+重命名+删除+置顶+面板互斥 | 🆕 P4 |
| **Cell 服务端自动保存** | ChatGPT 模式: 用户消息+NaNaGi回复由route.ts自动写入Cell | 🆕 P4 |
| **guest 记忆管理** | guest 可查看/删除自己的记忆 (MemoryPanel + LevelDB) | 🆕 P4 |
| **面板互斥系统** | Cell/Memory 面板共用 openPanel 状态，打开一个自动收另一个 | 🆕 P4 |

⚠️ IP 地理定位: 本地开发 IP 127.0.0.1 → 使用测试地点(北京)。部署后真实外网 IP 即可工作。
✅ Guest 用户细分: P3 已解决 — 每个注册用户有独立 personId 和 IWM Node 子目录。

### P1-P4 已修复 Bug 记录 (18条)

| Bug | 现象 | 根因 | 修复 |
|-----|------|------|------|
| 记忆泄露 | guest 对话中 NaNaGi 称对方"主人"，显示历史记忆 | buildMemoryContext 和 MemoryPanel API 未按 personId 过滤 | System Prompt + GET /api/memory + [slug] 均按 role 隔离 |
| Windows 文件名 crash | `ENOENT: open .../2026-06-07T02:59:21.178Z.json` | ISO 时间戳含冒号 `:`，Windows 文件名非法 | 所有时间戳 `:` 替换为 `-` |
| JWT 残留登录 | 删用户数据后刷新页面仍显示已登录 | cookie 未清 + GET /api/auth 未验证用户存在 | GET /api/auth 查数据库验证用户存在，不存在则删 cookie |
| 邮箱索引丢失 | 注册后 `_index/email/` 未生成 | store.ts 调旧 raw `dbPut` 绕过 emailIndexDir | store.ts 转发到 leveldb.ts 的 `putEmailIndex` |
| 环境 API 滥用 | 每次对话都调 ip-api + 和风 | 无节制调用 | 24h 全查询缓存 (时间<1ms, 地点+天气每天一次) |
| 密码冲突 | guest 密码与 admin 密码相同时误登 admin | `!personId` 条件未区分 email 登录 | 加 `!email` 判断——带邮箱的一定是 guest |
| DeepSeek 400 | `role "tool"` 不支持 / `content: null` 非法 | Anthropic 端点的 tool_result 格式不同 | role: "user" + tool_result JSON / content: "" |
| DeepSeek 400 (第二轮) | tool_result 注入后第二轮调用报 400 | `content: null` 在 assistant tool_use 消息中不合法 | assistant 带 tool_calls 时 content 改为 `""` |
| Cell 消息不持久化 | 刷新后聊天记录消失 | 服务端存 Cell 时 `agentLoop` 文本回复只通过 SSE 推送，未写回 `messages` 数组。`finalMessages.filter(m => m.role === "assistant" && m.content)` 返回空 | 在 `agentLoop` 文本回复分支加 `messages.push({ role: "assistant", content })` |
| Cell 消息重复 | 同一条消息出现多次 | 去重用了内容匹配——用户说两次"好的"被吞一条 | 改用消息 ID 去重 (PUT /api/cells/[cellId]/messages) |
| Memory API 403 | guest 点击记忆卡片报 403 | guest 调 `GET /api/memory/[slug]` 访问 admin 文件记忆 | guest 直接显示列表已有的 content，不请求 slug 端点。MemoryBrief 加 `content?` 字段 |
| Cell 面板互斥 | 记忆面板和对话面板同时打开互相遮挡 | 两个面板各自管理 `open` 状态 | ChatContext 加 `openPanel: "memory"|"cell"|null`，两面板共用，互斥 |
| Cell 面板点击自动收起 | 点 Cell 或"新对话"后面板消失 | `onClick` 里调了 `setOpenPanel(null)` | 移除 select/new 回调中的面板关闭逻辑 |
| 重命名编辑框体验差 | onBlur 自动提交 vs 点击取消冲突，确认按钮不明显 | onClick 在 onBlur 之后触发，按钮事件被吞 | 用 onMouseDown 替代 onClick（优先于 onBlur），加 ✓确认/✗取消 双按钮 |
| 邮箱验证码未传到后端 | 注册报 400 | AuthForm 的 `handleRegister` 调 `onLogin` 时没传 `code` 参数 | 补充 `regCode` 参数到 `onLogin` 调用链 |
| 用户删除数据后仍保持登录 | 删数据刷新仍显示已登录 | JWT cookie 未清 + `GET /api/auth` 未验证用户存在 | 查数据库验证用户，不存在则 `cookieStore.delete()` |
| Windows 文件名冒号 | ISO 时间戳 `2026-06-07T02:59:21.178Z` 写文件 crash | NTFS 不允许文件名含 `:` | 所有 LevelDB 时间戳 `:` 替换为 `-` |
| Email 索引丢失 | 注册后 `_index/email/` 未生成 | `store.ts` 调 `dbPut("email:...")` 绕过了 `emailIndexDir()` | `store.ts` 转发到 `leveldb.ts` 的 `putEmailIndex` |
| 环境 API 滥用 | 每次 POST /api/chat 都调 ip-api + 和风 | 无节制调用 | 24h 全查询缓存 (时间<1ms, 地点+天气每天一次) |
| Outlook SMTP 认证失败 | 535 5.7.139 Authentication unsuccessful | 微软2022年禁了个人邮箱SMTP | 换 QQ 邮箱 SMTP (smtp.qq.com:465 + 授权码) |
| 和风天气 403 | 天气 API 返回 403 Invalid Host | API Key 需放 HTTP Header `X-QW-Api-Key`，Host 需用专属域名 | Key 移到 Header, Host 改为 `mt7fc3reav.re.qweatherapi.com` |
| classic-level 编译失败 | Turbopack 报 `No native build was found` | native C++ 模块与 Turbopack 不兼容 | 换文件系统 K-V 实现，接口不变 |

### 路线图

| 阶段 | 内容 | 对应 JD 职责 | 预计 |
|------|------|-------------|------|
| **P1** | ✅ Agent Engine: ReAct + 工具 + 容灾 + 环境感知 + route.ts 重构 | 1/2/3 | 完成 |
| **P2** | ✅ Storage Foundation: store.ts + 六表K-V + IWM持久化 | 4 | 完成 |
| **P3** | ✅ 注册登录: 邮箱验证码 + personId分化 + IWM Cold Start + 数据隔离 | 1 | 完成 |
| **P4** | ✅ Cell 系统: 对话列表+滑出面板+持久化+重命名+删除+置顶+面板互斥 | 1 | 完成 |
| **P5** | 数字人格引擎: 社交图 + OCC + AmbientContext + SIP + 内心独白 | 4/5 | 进行中 |
| **P6** | 面试反馈: 反馈记录 + search_guests/get_feedback + MemoryPanel改造 | 1/5 | 待 P5 |
| **P7** | 知识库: 展厅结构化数据填充 | 1 | 待 P6 |
| **P8** | 上线部署 + 性能量化 + 博客 + demo | 3/5/6 | 待 P7 |

---

## 字节 JD 差距审计

> 对照：字节跳动 Agent应用开发工程师-ArkClaw (A156568) 六条职责
> 审计日期：2026-06-08 (基于 P1-P4 完成状态)

### 覆盖度总览 (更新)

```
职责 1 (架构/Skills):  ██████░░ 60%  ← +20% (ReAct+注册表+Cell)
职责 2 (任务规划):      ██░░░░░░ 20%  ← 不变 (SIP 仍为设计)
职责 3 (性能/容灾):     █████░░░ 50%  ← +20% (三层容灾+环境缓存)
职责 4 (框架沉淀):      ████░░░░ 40%  ← +20% (agent/模块化+统一数据层)
职责 5 (前沿跟踪):      ████████░ 80%  ← 不变
职责 6 (上线落地):      ███░░░░░ 30%  ← 不变 (仍未部署)

综合: ~47% (↑ 从 37%)
```

### 逐条审计 (更新)

**职责 1 — Agent 应用整体架构设计与核心功能开发**

| 子项 | 现状 | 评级 |
|------|------|------|
| 大模型适配 | DeepSeek V4 Pro 单模型。三层容灾已实现(30s超时→重试→降级)。无多模型切换/key rotation | 🟡 |
| 记忆库 | 双路径记忆(模型主动+框架兜底) + 按personId过滤 + guest可删除 | 🟢 |
| **知识库** | 3 个项目展厅骨架。get-project-info 工具可返回结构化数据。展厅内容未深度填充 | 🟡 |
| Skills 系统 | ✅ 8 个工具, Map注册表, ReAct 5轮循环, hash循环检测, 标准 execute 接口 | 🟢 |

**职责 2 — 任务规划、逻辑推理、决策执行**

| 子项 | 现状 | 评级 |
|------|------|------|
| **任务规划** | SIP 六步 + Gross 五策略设计完整。P5 实现。当前 agentLoop 为纯 ReAct 无规划层 | 🔴 |
| 逻辑推理 | 依赖 DeepSeek 自身。ReAct 循环提供了框架级多步推理基础 | 🟡 |
| 决策执行 | ✅ ReAct 5轮 + hash循环检测 + 三层容灾 + 8工具并行执行 | 🟢 |

**职责 3 — 性能优化与稳定性保障**

| 子项 | 现状 | 评级 |
|------|------|------|
| 响应延迟 | ✅ SSE 流式。环境感知 24h 全查询缓存 (时间<1ms, 地点+天气每天1次) | 🟢 |
| **并发能力** | 无任何设计 | 🔴 |
| 错误容灾 | ✅ 三层降级: 30s超时→重试1次→降级回复(因role措辞不同) | 🟢 |

**职责 4 — 沉淀通用 Agent 开发框架**

| 子项 | 现状 | 评级 |
|------|------|------|
| 框架解耦 | ✅ agent/ 独立模块(types/registry/loop/prompts/tools)。lib/store.ts 统一数据接口, 换存储只改 leveldb.ts | 🟢 |
| 组件库 | AgentDialog/ChatMessage/MemoryPanel/CellList 可复用但未抽象为独立库 | 🟡 |
| 技术方案 | ✅ 完整架构设计 + 19学术引用 + GNN映射 + JD审计 + 完整文件清单 | 🟢 |

**职责 5 — 跟踪 Agent 领域前沿技术**

| 子项 | 现状 | 评级 |
|------|------|------|
| 学术跟踪 | ✅ 19引用 (Bowlby→GraphSAGE), 从发展心理学到图神经网络 | 🟢 |
| 竞品分析 | ✅ Open-AGC 完整评测 (代码/架构/功能/UI 全维度对比) | 🟢 |
| **MCP / A2A** | 不涉及 | 🟡 |
| **技术博客** | 未写 | 🟡 |

**职责 6 — 从需求到上线的全流程落地**

| 子项 | 现状 | 评级 |
|------|------|------|
| **上线部署** | 未部署 | 🔴 |
| 版本迭代 | ✅ V1→V2→V2.5→v5.1 清晰演进记录 | 🟢 |
| UI/产品体验 | ✅ 三风格设计 + 唱片机 + Cell滑出面板 + 邮箱验证 + 注册引导 | 🟢 |
| **用户反馈闭环** | 无 | 🟡 |

### 当前短板 (更新)

**1. 知识库未填充 (影响职责 1)**
三个展厅(P7)有骨架无内容。面试官问"你主人做了什么"→ get_project_info 能回答, 但不够深入。→ P7

**2. 任务规划层不存在 (影响职责 2)**
SIP 六步是完整设计但代码为零。当前 ReAct 是纯响应式——无预先规划。→ P5-6

**3. 社交图是最强牌但未集成 (影响面试叙事)**
P5 完成后, 社交图+OCC+AmbientContext+内心独白会成为 JD 评分表外的核武器。→ P5

---

## 历史规划存档 (Block×P & P0-P4)

> ⚠️ **存档说明**: 以下为 2026-06-07 制定的 Block×P 映射与 P0-P4 优先级路线。保留作为设计演进的历史参考，不再作为当前开发依据。当前实施计划见 [实施计划 v5.1](#实施计划-v51-phase-1-8-当前)。

---

## 实施计划 v5.1 (Phase 1-8, 当前)

## 实施计划 v5.1：8 阶段 × 31 任务

### Phase 1: Agent Engine `🔴 当前 — 最低面试门槛`

让娜娜吉成为真正的 Agent：ReAct 多轮循环、9 工具注册表、三层容灾、双通道 System Prompt。

```
#23 P1-1 agent/types.ts — Agent 核心类型 (60行)
       │
       ├── #24 P1-2 personality/configs/ — 双通道参数 (120行)
       │         │
       ├── #25 P1-3 agent/registry.ts — 工具注册表 (40行)
       │         │
       │         └── #26 P1-4 agent/tools/ — 9个工具 (350行)
       │                    │
       │                    └──────┐
       │                           │
       └── ── ── ── ── ── ── ── ──│─ ── ─┘
                                   │
                            #27 P1-5 agent/prompts.ts — 七段式Prompt (150行)
                                   │
                            ┌──────┘
                            ▼
                     #28 P1-6 agent/loop.ts — ReAct循环+三层容灾 (200行)
                            │
                            ▼
                     #29 P1-7 middleware + route.ts — 集成 (644→80行)

              #54 P1-8 lib/geo.ts — IP地理定位 (40行) [独立, P1-7前完成]
```

| 任务 | 文件 | 行数 | 产出 |
|------|------|------|------|
| P1-1 | `agent/types.ts` | 60 | AgentMessage, ToolCall, AgentContext 等核心类型 |
| P1-2 | `personality/configs/{guest,admin}.ts` | 120 | 角色描述/称呼/钳制/目标/策略/行为准则 |
| P1-3 | `agent/registry.ts` | 40 | register/get/list/execute — 工具注册表 |
| P1-4 | `agent/tools/*.ts` (9文件) | 350 | 5提取+4新增，每个工具有 schema+execute |
| P1-5 | `agent/prompts.ts` | 150 | buildSystemPrompt(role) — 七段式拼接 |
| P1-6 | `agent/loop.ts` | 200 | ReAct 5轮+hash循环检测+30s超时→重试→降级 |
| P1-7 | `middleware.ts` + `route.ts` | +5/-564 | role/personId 注入 + 薄层 handler |
| P1-8 | `lib/geo.ts` | 40 | request.ip → geoip-lite → {city, country, coords} |

**Phase 1 产出**: 娜娜吉能动 — ReAct 多轮、七段式 Prompt、guest/admin 双通道、IP 地理定位。面试官打开网站看到一个真正的 Agent。

**Phase 1 状态**: ✅ 完成 (2026-06-07)
- 17 个新文件, ~1,100 行代码
- TypeScript 零编译错误
- DeepSeek ReAct 循环跑通 (5 轮 + hash 循环检测 + 三层容灾)
- 双通道 System Prompt 工作 (admin: 主人通道 / guest: 面试官通道)
- 时间感知正常 (7段 + 季节 + 节假日)
- 天气自动获取正常 (和风 API, 1h 缓存, X-QW-Api-Key Header 认证)
- 8 个工具注册到 registry (search-web 场景不需要, 已移除)
- API 安全收敛: lib/env.ts 集中管控 + import "server-only" + middleware Edge 兼容
- route.ts 644→160 行重构完成
- ⚠️ IP 地理定位: 本地 127.0.0.1 无法定位 → 部署后真实外网 IP 即可工作

### Phase 2: Storage Foundation

```
#30 P2-1 lib/store.ts → #31 P2-2 LevelDB初始化 → #32 P2-3 Loop接入存储
```

| 任务 | 文件 | 行数 | 产出 |
|------|------|------|------|
| P2-1 | `lib/store.ts` | 120 | 统一数据接口 — admin→fs, guest→LevelDB |
| P2-2 | `lib/leveldb.ts` + `data/` | 60 | classic-level 六表建库 (user/iwm/mem/emo/conv/feedback) |
| P2-3 | `agent/loop.ts` 改造 | 80 | Step 10 后处理接入 store.ts |

**Phase 2 产出**: 三层分级存储落地 — NaNaGi 本体文件系统 / 主人文件系统 / guest LevelDB。

### Phase 3: Auth & Registration

```
#33 P3-1 注册API+IWM初始化 → #34 P3-2 登录改造 → #35 P3-3 前端UI
```

| 任务 | 文件 | 行数 | 产出 |
|------|------|------|------|
| P3-1 | `app/api/auth/register/` + `personality/iwm-init.ts` | 60 | 引导式注册 → personId + IWM Cold Start |
| P3-2 | `app/api/auth/route.ts` | 40 | personId+password → O(1) LevelDB 查 → JWT |
| P3-3 | `components/AuthForm.tsx` + `AgentDialog.tsx` | 120 | 三步注册表单 + 登录界面 |

**Phase 3 产出**: 注册即 Cold Start — 表单直接映射 IWM Node 初始化。

### Phase 4: Cell System

```
#36 P4-1 Cell模型+CRUD → #37 P4-2 Cell列表前端 → #38 P4-3 Cell切换+持久化
```

| 任务 | 文件 | 行数 | 产出 |
|------|------|------|------|
| P4-1 | `lib/cell-store.ts` | 100 | CellRecord(messages[]) + LevelDB conv 表 CRUD |
| P4-2 | `components/CellList.tsx` | 100 | 左侧 Cell 列表 — 按时间倒序, 点击切换 |
| P4-3 | `contexts/ChatContext.tsx` 改造 | 120 | cellId 状态管理 + 消息自动持久化 |

**Phase 4 产出**: ChatGPT 同款 Cell 隔离 — 完整消息存储、跨 Cell 上下文隔离、滑出面板、置顶/重命名/删除、面板互斥。

**Phase 4 状态**: ✅ 完成 (2026-06-08)
- 新增 5 个 API 端点 (cells CRUD + cell messages)
- Cell 滑出面板 (跟 MemoryPanel 同款设计, 互斥)
- 服务端自动消息持久化 (ChatGPT 模式)
- guest 记忆管理 (查看/删除自己的记忆)
- Cell 置顶/重命名/删除
- 17 个 Bug 修复

### Phase 5: Personality Engine `🔥 差异化核心`

```
#39 P5-1 personality/types.ts (180行)
         │
         ├── #40 P5-2 graph.ts — 社交图引擎 (150行)
         ├── #41 P5-3 emotion.ts — OCC情绪引擎 (150行)
         ├── #42 P5-4 ambient-context.ts — 环境感知 (120行)
         ├── #43 P5-5 signals.ts — 外部信号提取 (100行)
         ├── #44 P5-6 planning.ts — SIP社交规划 (120行)
         ├── #45 P5-7 filter.ts — 人格过滤层 (80行)
         ├── #46 P5-8 inner-voice.ts — 内心独白 (100行)
         ├── #47 P5-9 memory-inner.ts — 隐形记忆 (60行)
         │
         └──→ #48 P5-10 人格引擎接入Agent Loop (80行)
```

| 任务 | 文件 | 行数 | 学术引用 |
|------|------|------|---------|
| P5-1 | `personality/types.ts` | 180 | SelfNode[15], IWMNode[1][3], EmotionState[6][7], AmbientContext, PersonaParameters[5], SocialPlan[11] |
| P5-2 | `personality/graph.ts` | 150 | Bowlby IWM[1][2][3] + Heider[4] + GraphSAGE[19] |
| P5-3 | `personality/emotion.ts` | 150 | OCC[8] + LeDoux[9] + McEwen[10] + Gross[12] |
| P5-4 | `personality/ambient-context.ts` | 120 | PAD[6] + Plutchik[7] |
| P5-5 | `personality/signals.ts` | 100 | ToM[13] — 确定性规则引擎, 不经 LLM |
| P5-6 | `personality/planning.ts` | 120 | Crick&Dodge SIP[11] + Gross[12] |
| P5-7 | `personality/filter.ts` | 80 | Jung Persona[5] — 确定性映射, 不经 LLM |
| P5-8 | `personality/inner-voice.ts` | 100 | LeDoux 高通路[9] |
| P5-9 | `personality/memory-inner.ts` | 60 | — |
| P5-10 | `agent/loop.ts` + `agent/prompts.ts` 改造 | 80 | 全引擎接入: Step1-8 串联 |

**Phase 5 产出**: 完整数字人格 — 社交图 + 情绪引擎 + 环境感知 + 社交规划 + 内心独白。

### Phase 6: Interview Feedback

| 任务 | 文件 | 行数 | 产出 |
|------|------|------|------|
| P6-1 | `agent/loop.ts` 改造 | 60 | guest-iv 对话结束 → 检测评价/岗位信号 → 追加 feedback |
| P6-2 | `agent/tools/search-guests.ts` + `get-feedback.ts` | 80 | admin 专属工具: 列出最近访客 + 查看反馈 |
| P6-3 | `components/MemoryPanel.tsx` 改造 | 100 | 新增「💭 内心」Tab + 「📊 反馈」Tab (仅 admin) |

### Phase 7: Knowledge Base

| 任务 | 文件 | 行数 | 产出 |
|------|------|------|------|
| P7 | `lib/projects.ts` + 展厅组件 | 200 | 三个展厅结构化数据 — Agent 能回答项目问题 |

### Phase 8: Deployment & Polish

| 任务 | 产出 |
|------|------|
| P8 | 性能量化(TTFT+延迟+成功率) + 并发设计文档 + 腾讯云部署 + 技术博客 + 双通道对比demo |

### 依赖链总览

```
P1 (Agent Engine) ──→ P2 (Storage) ──→ P3 (Auth) ──→ P4 (Cell)
                         │                                 │
                         └──→ P5 (Personality) ←───────────┘
                                  │
                                  └──→ P6 (Feedback)

P7 (Knowledge Base) ← 独立，随时可做
P8 (Deploy) ← P1 完成后即可开始准备
```

### 总规模估算

| Phase | 任务数 | 新代码(行) | 重构(行) | 累计(行) |
|-------|--------|-----------|----------|---------|
| P1: Agent Engine | 8 | ~980 | -564 | ~1,540 |
| P2: Storage | 3 | ~260 | ~80 | ~1,840 |
| P3: Auth | 3 | ~220 | — | ~2,060 |
| P4: Cell | 3 | ~320 | ~120 | ~2,500 |
| P5: Personality | 10 | ~1,140 | ~80 | ~3,720 |
| P6: Feedback | 3 | ~240 | — | ~3,960 |
| P7: Knowledge | 1 | ~200 | — | ~4,160 |
| P8: Deploy | 1 | — | — | — |
| **总计** | **32** | **~3,360** | **~120** | **~4,200** |

---

## 文件结构

```
D:/NanAgi/
│
├── README.md                       ← 本文档 (完整架构设计)
│
├── src/                            ← ✅ 已实现 / 📋 规划中 / 🔜 进行中
│
│   ├── agent/                      ✅ P1: Agent 机械层
│   │   ├── types.ts                ✅ AgentMessage, ToolCall, AgentContext 等
│   │   ├── registry.ts             ✅ ToolRegistry { register, get, list, execute }
│   │   ├── loop.ts                 ✅ ReAct 循环 (5轮+hash检测+三层容灾)
│   │   ├── prompts.ts              ✅ 七段式 System Prompt
│   │   └── tools/                  ✅ 8 个工具 (search-web 已移除以简化场景)
│   │       ├── index.ts, get-time.ts, get-weather.ts
│   │       ├── get-project-info.ts, search-memory.ts, save-memory.ts
│   │       ├── generate-image.ts, navigate-project.ts
│   │       └── gnn-recommend.ts, cnnmusic-search.ts
│   │
│   ├── personality/                🔜 P1(configs) + 📋 P5(引擎)
│   │   ├── configs/                ✅ P1-2
│   │   │   ├── guest.ts            ✅ 面试官通道参数 (情绪钳制+预设目标+策略)
│   │   │   └── admin.ts            ✅ 主人通道参数 (0义务+拒绝权+真实表达)
│   │   ├── types.ts                📋 P5-1: SelfNode, IWMNode, GraphState, ...
│   │   ├── graph.ts                📋 P5-2: 社交图引擎
│   │   ├── emotion.ts              📋 P5-3: OCC评价 + 双弹簧
│   │   ├── ambient-context.ts      📋 P5-4: 时间·地点·天气 (P1已在lib/ambient.ts实现)
│   │   ├── signals.ts              📋 P5-5: 外部信号提取
│   │   ├── planning.ts             📋 P5-6: SIP社交规划
│   │   ├── filter.ts               📋 P5-7: 人格过滤层
│   │   ├── inner-voice.ts          📋 P5-8: 内心独白
│   │   ├── memory-inner.ts         📋 P5-9: 隐形记忆
│   │   └── iwm-init.ts             🔜 P3: 注册表单 → IWM Node 初始化
│   │
│   ├── lib/
│   │   ├── types.ts                ✅ SSE 事件 + Message + JukeboxState
│   │   ├── auth.ts                 ✅ bcrypt + JWT (P3 改造)
│   │   ├── memory.ts               ✅ 显性记忆 CRUD (文件系统)
│   │   ├── hunyuan.ts              ✅ 混元 API 客户端
│   │   ├── projects.ts             ✅ 项目元数据 (P7 填充)
│   │   ├── env.ts                  ✅ P1: 安全配置中心 (所有API Key唯一出口)
│   │   ├── ambient.ts              ✅ P1: 时间+IP定位+天气 (1h缓存)
│   │   ├── store.ts                ✅ P2: 统一数据接口 (admin→fs, guest→K-V)
│   │   ├── leveldb.ts              ✅ P2: 文件K-V (六表, 零原生依赖)
│   │   └── cell-store.ts           📋 P4: Cell CRUD
│   │
│   ├── components/
│   │   ├── AgentDialog.tsx         ✅ 首页对话
│   │   ├── AgentWidget.tsx         ✅ 项目页浮动聊天
│   │   ├── ChatMessage.tsx         ✅ 消息气泡
│   │   ├── ChatInput.tsx           ✅ 输入框
│   │   ├── DraggableImage.tsx      ✅ 可拖拽图片
│   │   ├── RecordPlayer.tsx        ✅ 唱片机
│   │   ├── MemoryPanel.tsx         ✅ 记忆面板 (P6 加 Tab)
│   │   ├── PageShell.tsx           ✅ 布局壳
│   │   ├── AuthForm.tsx            🔜 P3: 注册/登录表单
│   │   └── CellList.tsx            📋 P4: Cell 列表
│   │
│   ├── contexts/
│   │   └── ChatContext.tsx         ✅ 聊天状态 (P4 Cell 改造)
│   │
│   ├── middleware.ts               ✅ JWT验证 + role/personId header注入
│   │
│   └── app/
│       ├── api/
│       │   ├── auth/route.ts       ✅ 登录 (P3 改造)
│       │   ├── auth/register/      🔜 P3: 注册端点
│       │   ├── chat/route.ts       ✅ P1 重构: 644→180行 thin handler
│       │   └── memory/             ✅ 记忆 CRUD
│       ├── projects/[slug]/        ✅ 3 个项目展厅
│       ├── layout.tsx, page.tsx    ✅
│       └── globals.css             ✅ 三风格设计系统
│
├── data/
│   ├── admin/                      ✅ P2: 南志锦专属
│   │   └── nanzhijin-iwm.json     ✅ IWM Node (每次对话自动更新)
│   ├── leveldb/                    ✅ P2: 文件K-V (guest用户)
│   │   └── iwm:{personId}.json    ⚠️ 所有guest共用 personId="guest" → 待P3分化
│   ├── memory/                     ✅ 显性记忆 (V2.5)
│   ├── self/                       📋 P5: NaNaGi 本体
│   └── inner/                      📋 P5: 内心独白
│
├── package.json
├── tsconfig.json
├── next.config.ts
├── tailwind.config.ts
├── Dockerfile
├── .env.example
└── .env.local (不提交)
```

---

## 完整文件清单 (P4 当前状态)

> 每个文件的职责分析，用于目录结构决策。

### `src/agent/` — Agent 机械层

| 文件 | 职责 | 依赖 |
|------|------|------|
| `types.ts` | Agent 核心类型: ToolCall, AgentContext, LoopState 等 | 无 |
| `registry.ts` | 工具注册表: Map<name, {definition, execute}> | types.ts |
| `loop.ts` | ReAct 循环引擎: 5轮+hash检测+三层容灾+DeepSeek SSE解析 | types, prompts, registry, lib/env |
| `prompts.ts` | System Prompt 七段式拼接: 角色/人格/关系/记忆/工具/准则/上下文 | types, registry, configs |
| `tools/index.ts` | 工具统一入口: import 即注册 | 所有工具文件 |
| `tools/get-time.ts` | 本地时间查询 | registry |
| `tools/get-weather.ts` | 和风天气查询 | registry, lib/env |
| `tools/get-project-info.ts` | 项目信息查询 (读 lib/projects.ts) | registry, lib/projects |
| `tools/search-memory.ts` | 显性记忆关键词搜索 | registry, lib/memory |
| `tools/save-memory.ts` | 主动记忆保存 (admin→fs, guest→LevelDB) | registry, lib/memory, lib/store |
| `tools/generate-image.ts` | 混元生图 + SSE jukebox | registry, lib/hunyuan |
| `tools/navigate-project.ts` | SSE navigate → 跳转展厅 | registry |
| `tools/gnn-recommend.ts` | GNN 推荐 (占位) | registry |
| `tools/cnnmusic-search.ts` | CnnMusic 检索 (占位) | registry |

### `src/personality/` — 数字人格层

| 文件 | 职责 | 依赖 |
|------|------|------|
| `configs/guest.ts` | 面试官通道参数: 角色/情绪钳制/预设目标/Gross策略/行为准则 | agent/types |
| `configs/admin.ts` | 主人通道参数: 0义务/拒绝权/真实表达 | agent/types |
| `iwm-init.ts` | 注册表单 → IWM Node 冷启动 | lib/leveldb |

### `src/lib/` — 共享基础设施

| 文件 | 职责 | 分类 |
|------|------|------|
| `types.ts` | SSE 事件类型 + Message + JukeboxState (前端 + 后端共用) | 类型 |
| `auth.ts` | JWT 签发/验证 + 双密码 bcrypt + Cookie 管理 | 🔐 鉴权 |
| `env.ts` | 安全配置中心: 所有 API Key 唯一出口, `import "server-only"` | 🔐 配置 |
| `email.ts` | SMTP 邮件发送 (nodemailer) + 验证码存储 (内存Map, 5min TTL) | 📧 邮件 |
| `ambient.ts` | 环境感知: 时间(7段+季节+节假日) + IP定位(ip-api) + 天气(和风,1h+24h缓存) | 🌤 环境 |
| `leveldb.ts` | 文件系统 K-V 存储引擎: 六表schema, 用户子目录隔离, 全局查询 | 💾 存储 |
| `store.ts` | 统一数据访问层: admin→文件系统, guest→LevelDB, 内部路由 | 💾 存储 |
| `memory.ts` | 显性记忆 CRUD: 文件系统 YAML frontmatter (V2.5, 保留给 admin) | 💾 存储 |
| `hunyuan.ts` | 腾讯混元生图 API: async submit + 轮询 query | 🎨 外部API |
| `projects.ts` | 项目元数据: 3个项目 (GNN/CnnMusic/FruitCNN) | 📋 数据 |

### `src/components/` — UI 组件

| 文件 | 职责 | 分类 |
|------|------|------|
| `AgentDialog.tsx` | 首页全屏对话: 登录/注册/聊天/Cell 集成 | 🖥 聊天 |
| `AgentWidget.tsx` | 项目页右下角浮动聊天 FAB | 🖥 聊天 |
| `ChatMessage.tsx` | 消息气泡: agent/user, 图片渲染, streaming动画 | 🖥 聊天 |
| `ChatInput.tsx` | 聊天输入框: Enter发送 | 🖥 聊天 |
| `AuthForm.tsx` | 引导式注册/登录: 三步注册+验证码+邮箱登录 | 🔐 鉴权 |
| `CellList.tsx` | 对话列表滑出面板: 置顶/重命名/删除/切换 | 📋 Cell |
| `MemoryPanel.tsx` | 记忆库滑出面板: 卡片列表/详情弹窗/删除 | 📋 记忆 |
| `PageShell.tsx` | 页面布局壳: usePathname 判断首页/项目页 | 📐 布局 |
| `RecordPlayer.tsx` | 唱片机: 拖拽识别动画 (项目展厅) | 🎵 项目 |
| `DraggableImage.tsx` | 可拖拽图片: Portal克隆+磁吸回弹 | 🎵 项目 |

### `src/contexts/` — React 状态管理

| 文件 | 职责 |
|------|------|
| `ChatContext.tsx` | 全局状态: 消息/auth/jukebox/modelResult/cell/memory/面板互斥 |

### `src/app/` — Next.js 路由

| 文件 | 职责 | 分类 |
|------|------|------|
| `api/auth/route.ts` | 登录: admin密码 / guest email+password | 🔐 鉴权 |
| `api/auth/register/route.ts` | 注册: 邮箱验证码+格式验证+一次性邮箱拦截+拼写纠错 | 🔐 鉴权 |
| `api/auth/send-code/route.ts` | 发送邮箱验证码 | 📧 邮件 |
| `api/chat/route.ts` | 聊天核心: 认证→环境感知→AgentContext→agentLoop→Cell持久化→IWM | 🤖 Agent |
| `api/cells/route.ts` | Cell CRUD: GET列表 / POST创建 | 📋 Cell |
| `api/cells/[cellId]/route.ts` | Cell 操作: PATCH重命名+置顶 / DELETE删除 | 📋 Cell |
| `api/cells/[cellId]/messages/route.ts` | Cell 消息: GET历史 / PUT保存 | 📋 Cell |
| `api/memory/route.ts` | 记忆 CRUD: GET列表(按role过滤) / POST创建 | 📋 记忆 |
| `api/memory/[slug]/route.ts` | 记忆操作: GET详情(仅admin) / DELETE(admin+guest自己) | 📋 记忆 |
| `layout.tsx` | 根布局: ChatProvider + PageShell | 📐 布局 |
| `page.tsx` | 首页: AgentDialog | 📐 布局 |
| `globals.css` | 三风格设计系统 + Memory/Cell 面板动画 | 🎨 样式 |
| `projects/[slug]/page.tsx` | 项目页 SSG | 🎵 项目 |
| `projects/[slug]/ProjectExhibition.tsx` | 项目展厅: 唱片机+指标卡 | 🎵 项目 |
| `projects/[slug]/ProjectPageClient.tsx` | 项目上下文注册 | 🎵 项目 |
| `middleware.ts` | JWT 验证 + role/personId header 注入 | 🔐 鉴权 |

### 分类统计

```
分类           文件数   目录
────────────────────────────────
🤖 Agent 引擎    14     agent/
🔐 鉴权          6      lib/auth, lib/env, middleware, api/auth/*(3), components/AuthForm
💾 存储          3      lib/{leveldb,store,memory}
📋 Cell         4      api/cells/*(3), components/CellList
📋 记忆         3      api/memory/*(2), components/MemoryPanel
📧 邮件          2      lib/email, api/auth/send-code
🌤 环境          1      lib/ambient
🎨 外部API       2      lib/{hunyuan,projects}
🖥 聊天          4      components/{AgentDialog,AgentWidget,ChatMessage,ChatInput}
🎵 项目          5      app/projects/*(3), components/{RecordPlayer,DraggableImage}
📐 布局          3      components/PageShell, app/{layout,page}
🎨 样式          1      globals.css
📋 数据          1      lib/projects
📋 类型          2      lib/types, agent/types
────────────────────────────────
总计            51      (含 middleware.ts + ChatContext.tsx)
```

---

## 技术栈

| 层 | 技术 |
|------|------|
| 前端框架 | Next.js 16 + TypeScript + React 19 |
| 样式 | Tailwind CSS 4 + 像素风三风格设计系统 |
| AI 引擎 | DeepSeek V4 Pro (Anthropic 兼容端点) |
| 鉴权 | bcryptjs + jose (JWT, personId + role) |
| 生图 | 腾讯混元 hy-image-v3.0 |
| 天气 + 地理 | 和风天气 API + geoip-lite (MaxMind GeoLite2) |
| 用户 IWM 存储 | LevelDB (classic-level, 嵌入式 K-V, MIT, O(1)) |
| 娜娜吉本体存储 | 文件系统 (JSON + Markdown, sub-ms, 可审计) |
| 显性记忆存储 | admin: 文件系统 / guest: LevelDB |
| 向量检索 (P2) | LanceDB (嵌入式, 零部署) |
| 部署 | Docker → 腾讯云 Lighthouse 2C4G 5M |

---

## 本地运行

```bash
npm install
cp .env.example .env.local  # 填入 API Key
npm run dev                  # http://localhost:3000
```

### 环境变量

| 变量 | 说明 |
|------|------|
| `NANAGI_ADMIN_PASSWORD_HASH` | 管理员密码 bcrypt hash |
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `HUNYUAN_API_KEY` | 混元生图 API Key |
| `WEATHER_API_KEY` | 和风天气 API Key |

---

## 面试话术

> "NaNaGi 不是工具型 Agent，是关系型 Agent [1]。核心架构是一个社交图——基于 Bowlby 内部工作模型 [1][2][3]，每个人在她心中有一个独立的 IWM 节点，6 个维度，弹性系数随关系深度变化 [10]。同一个锚定人格 (Self-Node) [15] 在不同社交情境中 (Jung 人格面具 [5]) 表现出不同的行为——面试官看到专业女仆，主人看到真实自我。
>
> 情绪不是 prompt 里的形容词，是独立的 OCC 评价引擎 [8]——外部信号驱动、规则引擎计算、不经 LLM 手、每一笔变化都有 audit log。高通路 [9] 在关键时刻触发内心独白，但不替她决定情绪。双弹簧拉回 [10] 确保情绪的弹性——不会因为一次对话就彻底改变。
>
> 存储架构是三层的——NaNaGi 本体用文件系统 (可审计、可 cat)，主人数据用文件系统 (可编辑)，所有 guest 用户数据走 LevelDB (O(1)、事务安全、百万级验证)。admin 和 guest 的 IWM Schema 完全一致——只有存储位置不同。
>
> Cell 隔离记忆架构确保上下文不膨胀——每个对话 Cell 是独立的上下文容器，IWM、显性记忆、面试反馈跨 Cell 累积，但其他 Cell 的完整聊天历史不进当前上下文。
>
> 两个图各自独立——Admin 通道的个人关系图用 Heider 平衡 [4] 维护熟人网络；Guest 通道的面试反馈图结构化记录面试官对南志锦的评价和需求，主人可事后查询。
>
> 这跟我做的 GNN 社交图谱链接预测是同一套数学框架 [19]——Node Embedding、Edge Weight、Message Passing、Cold Start。”

---

## License

Private — 仅供面试使用。源码不公开。
