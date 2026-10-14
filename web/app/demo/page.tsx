import type { Metadata } from "next";
import Link from "next/link";
import Demo from "@/components/Demo";

export const metadata: Metadata = {
  title: "RinkIQ · interactive demo",
  description: "Click around a hockey game that RinkIQ analysed: players, speeds, plays, report cards.",
  robots: { index: true, follow: true },
};

export default function DemoPage() {
  return (
    <>
      <div className="crumbs">
        <Link href="/">← RinkIQ</Link>
        <span className="hint" style={{ margin: 0 }}>No upload, no setup: this page is pre-analysed data</span>
      </div>
      <section className="board">
        <div className="brand">
          <h1>Interactive demo<span className="puck" /></h1>
        </div>
        <p className="tag" style={{ marginTop: 8, marginBottom: 14 }}>
          This is what RinkIQ produces from game film. Press play, <strong>click any skater</strong> to see their speed
          chart and report card, and drag the timeline. Everything shown was computed by the pipeline from video:
          positions on the rink in feet, speeds, body facing, plays and grades.
        </p>
        <Demo />
      </section>
    </>
  );
}
