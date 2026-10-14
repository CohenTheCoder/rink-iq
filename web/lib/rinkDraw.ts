export const RINK_L = 200;
export const RINK_W = 85;

/** Draw an NHL rink (200 × 85 ft) onto a canvas. Returns the feet → pixels mapping. */
export function drawRink(ctx: CanvasRenderingContext2D, width: number, height: number) {
  const m = Math.max(10, width * 0.012);
  const s = (width - 2 * m) / RINK_L;
  const X = (x: number) => m + x * s;
  const Y = (y: number) => height - m - y * s; // far boards at the top, like the broadcast camera
  ctx.clearRect(0, 0, width, height);
  const r = 28 * s;
  ctx.beginPath();
  ctx.roundRect(X(0), Y(RINK_W), RINK_L * s, RINK_W * s, r);
  ctx.fillStyle = "#f4f8fb";
  ctx.fill();
  ctx.save();
  ctx.clip();
  const line = (x: number, color: string, w: number) => {
    ctx.beginPath(); ctx.moveTo(X(x), Y(0)); ctx.lineTo(X(x), Y(RINK_W)); ctx.strokeStyle = color; ctx.lineWidth = w * s * 0.35; ctx.stroke();
  };
  line(11, "#d0393e", 1); line(189, "#d0393e", 1);           // goal lines
  line(75, "#2f6fd6", 12); line(125, "#2f6fd6", 12);          // blue lines
  line(100, "#d0393e", 12);                                   // center line
  const circle = (cx: number, cy: number, rad: number, color: string, w = 0.8) => {
    ctx.beginPath(); ctx.arc(X(cx), Y(cy), rad * s, 0, Math.PI * 2); ctx.strokeStyle = color; ctx.lineWidth = w * s * 0.5; ctx.stroke();
  };
  const dot = (cx: number, cy: number, rad: number, color: string) => {
    ctx.beginPath(); ctx.arc(X(cx), Y(cy), rad * s, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
  };
  for (const cx of [31, 169]) for (const cy of [20.5, 64.5]) { circle(cx, cy, 15, "#d0393e"); dot(cx, cy, 1, "#d0393e"); }
  circle(100, 42.5, 15, "#2f6fd6"); dot(100, 42.5, 1, "#2f6fd6");
  for (const cx of [80, 120]) for (const cy of [20.5, 64.5]) dot(cx, cy, 1, "#d0393e");
  for (const g of [11, 189]) {                                // creases
    const d = g < 100 ? 1 : -1;
    ctx.beginPath();
    ctx.moveTo(X(g), Y(42.5 - 4)); ctx.lineTo(X(g + d * 6), Y(42.5 - 4));
    ctx.quadraticCurveTo(X(g + d * 7), Y(42.5), X(g + d * 6), Y(42.5 + 4)); ctx.lineTo(X(g), Y(42.5 + 4));
    ctx.fillStyle = "rgba(47,111,214,0.15)"; ctx.fill();
    ctx.strokeStyle = "#d0393e"; ctx.lineWidth = s * 0.3; ctx.stroke();
  }
  ctx.restore();
  ctx.beginPath();
  ctx.roundRect(X(0), Y(RINK_W), RINK_L * s, RINK_W * s, r);
  ctx.strokeStyle = "#9aa7b4"; ctx.lineWidth = 2; ctx.stroke();
  return { X, Y, s };
}
