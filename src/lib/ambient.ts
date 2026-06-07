// ============================================================
// NaNaGi 环境感知 — P1-8+P1-9 (合并)
// 每次请求自动获取: 时间 + IP地理位置 + 天气
// 不经 LLM, 不经工具调用 — route.ts 调一次, 注入 AgentContext
// ============================================================

import type { NextRequest } from "next/server";
import type { AmbientSnapshot, TimeSnapshot, LocationSnapshot, WeatherSnapshot } from "@/agent/types";
import { getWeatherConfig } from "@/lib/env";

// IP 地理定位 — ip-api.com 免费 API
// 45次/分钟限制, 本地请求走缓存完全够用
// 无需本地数据文件, 无编译问题

// ==================== 时间 ====================

function getTimeContext(): TimeSnapshot {
  const now = new Date();
  const h = now.getHours();

  return {
    timeOfDay:
      h >= 4 && h < 6  ? "dawn" :
      h >= 6 && h < 9  ? "morning" :
      h >= 9 && h < 12 ? "forenoon" :
      h >= 12 && h < 17 ? "afternoon" :
      h >= 17 && h < 19 ? "evening" :
      h >= 19 && h < 23 ? "night" :
      "midnight",
    dayOfWeek: [0, 6].includes(now.getDay()) ? "weekend" : "weekday",
    season:
      now.getMonth() >= 2 && now.getMonth() <= 4 ? "spring" :
      now.getMonth() >= 5 && now.getMonth() <= 7 ? "summer" :
      now.getMonth() >= 8 && now.getMonth() <= 10 ? "autumn" :
      "winter",
    isHoliday: isHoliday(now),
    hoursSinceLastTalk: 0, // P5 接入 IWM 后填充
    isFirstMeeting: false, // P3 接入注册后判断
  };
}

// ==================== 节假日查表 (中国) ====================

const HOLIDAYS_2026: Record<string, string> = {
  "2026-01-01": "元旦",
  "2026-02-17": "春节除夕",
  "2026-02-18": "春节初一",
  "2026-02-19": "春节初二",
  "2026-02-20": "春节初三",
  "2026-02-21": "春节初四",
  "2026-02-22": "春节初五",
  "2026-02-23": "春节初六",
  "2026-04-05": "清明",
  "2026-05-01": "劳动节",
  "2026-05-31": "端午节",
  "2026-08-19": "七夕",
  "2026-09-25": "中秋节",
  "2026-10-01": "国庆",
};

function isHoliday(now: Date): boolean {
  const key = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
  return key in HOLIDAYS_2026;
}

// ==================== 地理位置 ====================

async function getLocation(request: NextRequest): Promise<LocationSnapshot | null> {
  const ip =
    request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ||
    request.headers.get("x-real-ip") ||
    "127.0.0.1";

  // 本地开发 IP → 使用测试地点 (北京)
  if (ip === "127.0.0.1" || ip === "::1" || ip === "localhost") {
    return {
      city: "北京",
      country: "中国",
      timezone: "Asia/Shanghai",
      coordinates: { lat: 39.9, lng: 116.4 },
    };
  }

  try {
    const res = await fetch(`http://ip-api.com/json/${ip}?fields=city,country,timezone,lat,lon`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) return null;

    const data = await res.json();
    if (!data.city) return null;

    return {
      city: data.city,
      country: data.country || "Unknown",
      timezone: data.timezone || "Asia/Shanghai",
      coordinates: { lat: data.lat, lng: data.lon },
    };
  } catch {
    return null;
  }
}

// ==================== 天气 (1h缓存) ====================

interface WeatherCache {
  coords: { lat: number; lng: number };
  timestamp: number;
  data: WeatherSnapshot;
}

let weatherCache: WeatherCache | null = null;
const WEATHER_GAP_MS = 60 * 60 * 1000; // 1 小时

function deriveSunlight(
  timeOfDay: TimeSnapshot["timeOfDay"],
  condition: string
): WeatherSnapshot["sunlight"] {
  // 夜间 → none
  if (["midnight", "dawn"].includes(timeOfDay)) return "none";
  if (timeOfDay === "night") return "none";
  if (timeOfDay === "evening") return "low";
  // 白天 → 看天气
  if (["clear"].includes(condition)) return "high";
  if (["overcast", "fog", "drizzle"].includes(condition)) return "low";
  if (["rain", "snow", "storm"].includes(condition)) return "none";
  return "medium";
}

async function fetchWeather(coords: { lat: number; lng: number }): Promise<WeatherSnapshot | null> {
  const { apiKey, apiHost } = getWeatherConfig();
  if (!apiKey) return null;

  const loc = `${coords.lng.toFixed(2)},${coords.lat.toFixed(2)}`;
  const url = `${apiHost}?location=${loc}`;

  try {
    const res = await fetch(url, {
      signal: AbortSignal.timeout(5000),
      headers: { "X-QW-Api-Key": apiKey },
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      console.warn(`[Weather] ${res.status}: ${body.slice(0, 200)}`);
      return null;
    }

    const json = await res.json();
    if (json.code !== "200") {
      console.warn(`[Weather] API error code=${json.code}: ${JSON.stringify(json).slice(0, 200)}`);
      return null;
    }

    const now = json.now;
    if (!now) return null;

    const condition = now.text as string;
    const timeOfDay = getTimeContext().timeOfDay;

    return {
      condition: mapCondition(condition),
      temperature: Number(now.temp) || 0,
      humidity: Number(now.humidity) || 50,
      windSpeed: Number(now.windSpeed) || 0,
      sunlight: deriveSunlight(timeOfDay, condition),
    };
  } catch {
    return null;
  }
}

function mapCondition(text: string): WeatherSnapshot["condition"] {
  const t = text.toLowerCase();
  if (t.includes("雨") || t.includes("rain")) return t.includes("暴") || t.includes("大") ? "storm" : t.includes("小") || t.includes("细") ? "drizzle" : "rain";
  if (t.includes("雪") || t.includes("snow")) return "snow";
  if (t.includes("晴") || t.includes("clear") || t.includes("sun")) return "clear";
  if (t.includes("阴") || t.includes("云") || t.includes("cloud") || t.includes("overcast")) return "overcast";
  if (t.includes("雾") || t.includes("fog") || t.includes("霾") || t.includes("haze")) return "fog";
  return "clear";
}

async function getWeather(location: LocationSnapshot | null): Promise<WeatherSnapshot | null> {
  if (!location) return null;

  const now = Date.now();

  // 同坐标 + 1h 内 → 不查
  if (
    weatherCache &&
    weatherCache.coords.lat === location.coordinates.lat &&
    weatherCache.coords.lng === location.coordinates.lng &&
    now - weatherCache.timestamp < WEATHER_GAP_MS
  ) {
    return weatherCache.data;
  }

  // 超时或坐标变了 → 查, 失败用旧数据兜底
  const data = await fetchWeather(location.coordinates);
  if (data) {
    weatherCache = { coords: location.coordinates, timestamp: now, data };
    return data;
  }

  return weatherCache?.data ?? null; // API 挂了 → 旧数据兜底
}

// ==================== 统一入口 ====================

// ==================== 全查询节制 (24h) ====================
// 时间每次必拿 (<1ms, 零成本)
// 地点+天气每天只查一次 — 减少 ip-api + 和风 API 调用

interface FullQueryCache {
  timestamp: number;
  location: LocationSnapshot | null;
  weather: WeatherSnapshot | null;
}

let fullQueryCache: FullQueryCache | null = null;
const FULL_QUERY_GAP_MS = 24 * 60 * 60 * 1000; // 24 小时

/** 获取环境快照 — route.ts 每次请求调一次 */
export async function getAmbient(request: NextRequest): Promise<AmbientSnapshot> {
  const time = getTimeContext();
  const now = Date.now();

  // 24h 内有全查询记录 → 复用缓存, 跳过外部 API
  if (fullQueryCache && (now - fullQueryCache.timestamp) < FULL_QUERY_GAP_MS) {
    return { time, location: fullQueryCache.location, weather: fullQueryCache.weather };
  }

  // 超过 24h → 执行全查询
  const location = await getLocation(request);
  const weather = await getWeather(location);

  fullQueryCache = { timestamp: now, location, weather };

  return { time, location, weather };
}
