import { NextResponse, type NextRequest } from "next/server";

/**
 * One shared password for the whole site (HTTP Basic auth: any username, the password is RINKIQ_PASSWORD).
 * In production the site stays LOCKED until RINKIQ_PASSWORD is set, so a fresh deploy never exposes uploads
 * to the internet by accident.
 */
export function middleware(req: NextRequest) {
  const password = process.env.RINKIQ_PASSWORD;
  if (!password) {
    if (process.env.NODE_ENV !== "production") return NextResponse.next(); // local dev: open
    return new NextResponse(
      "RinkIQ is locked. Set the RINKIQ_PASSWORD environment variable (Vercel → Project → Settings → Environment Variables), then redeploy.",
      { status: 503, headers: { "content-type": "text/plain; charset=utf-8" } },
    );
  }
  const header = req.headers.get("authorization") || "";
  if (header.startsWith("Basic ")) {
    try {
      const decoded = atob(header.slice(6));
      const given = decoded.slice(decoded.indexOf(":") + 1);
      if (given.length === password.length) {
        let diff = 0;
        for (let i = 0; i < given.length; i++) diff |= given.charCodeAt(i) ^ password.charCodeAt(i);
        if (diff === 0) return NextResponse.next();
      }
    } catch {
      /* fall through to the prompt */
    }
  }
  return new NextResponse("Password required.", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="RinkIQ", charset="UTF-8"' },
  });
}

export const config = { matcher: ["/((?!_next/static|_next/image|favicon.ico|icon.svg).*)"] };
