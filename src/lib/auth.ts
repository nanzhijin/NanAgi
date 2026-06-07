import { SignJWT, jwtVerify } from "jose";
import { cookies } from "next/headers";
import bcrypt from "bcryptjs";
import { getPasswordHashes, getJwtSecret, getNodeEnv } from "@/lib/env";

export type UserRole = "guest" | "admin" | "guest-iv";

const SECRET = getJwtSecret();
const COOKIE_NAME = "nanagi_token";
const EXPIRES_IN = "1h";

// ==================== 密码验证 ====================

const HASHES = getPasswordHashes();

export function verifyPassword(password: string): { valid: boolean; role: UserRole } {
  if (HASHES.admin && bcrypt.compareSync(password, HASHES.admin)) {
    return { valid: true, role: "admin" };
  }
  if (HASHES.guest && bcrypt.compareSync(password, HASHES.guest)) {
    return { valid: true, role: "guest" };
  }
  return { valid: false, role: "guest" };
}

// ==================== JWT ====================

export interface TokenPayload {
  personId: string;
  role: UserRole;
  name: string;
  identity: string;
}

export async function createToken(
  personId: string,
  role: UserRole,
  name: string,
  identity: string
): Promise<string> {
  return new SignJWT({ sub: "nanagi_user", role, personId, name, identity })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime(EXPIRES_IN)
    .sign(SECRET);
}

export async function verifyToken(token: string): Promise<{ valid: boolean } & TokenPayload> {
  try {
    const { payload } = await jwtVerify(token, SECRET);
    return {
      valid: true,
      personId: (payload.personId as string) || payload.role as string,
      role: (payload.role as UserRole) || "guest",
      name: (payload.name as string) || "客人",
      identity: (payload.identity as string) || "面试官",
    };
  } catch {
    return {
      valid: false,
      personId: "guest",
      role: "guest",
      name: "客人",
      identity: "面试官",
    };
  }
}

export async function setAuthCookie(token: string): Promise<void> {
  const cookieStore = await cookies();
  cookieStore.set(COOKIE_NAME, token, {
    httpOnly: true,
    secure: getNodeEnv() === "production",
    sameSite: "lax",
    path: "/",
    maxAge: 3600,
  });
}

export async function getAuthCookie(): Promise<string | undefined> {
  const cookieStore = await cookies();
  return cookieStore.get(COOKIE_NAME)?.value;
}
