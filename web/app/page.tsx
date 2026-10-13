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
        <Uploader />
      </section>
      <JobList />
    </>
  );
}
