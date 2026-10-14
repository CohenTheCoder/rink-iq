import Link from "next/link";
import JobList from "@/components/JobList";
import SetupBanner from "@/components/SetupBanner";
import Uploader from "@/components/Uploader";

export default function Home() {
  return (
    <>
      <SetupBanner />
      <section className="board">
        <div className="brand">
          <h1>RinkIQ<span className="puck" /></h1>
          <span className="tag">Game film in. <span className="led">Stats</span> out.</span>
        </div>
        <p className="tag" style={{ marginTop: 8 }}>
          Upload hockey video from an elevated camera. RinkIQ finds the rink on its own, tracks every skater, reads jersey
          numbers, grades the players and draws it all back onto the film.
        </p>
        <p className="row" style={{ marginTop: 14 }}>
          <Link className="btn blue" href="/demo">Try the interactive demo</Link>
          <span className="hint" style={{ margin: 0 }}>No upload needed: click around a game RinkIQ already analysed.</span>
        </p>
        <Uploader />
      </section>
      <JobList />
    </>
  );
}
