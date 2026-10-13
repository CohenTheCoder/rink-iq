export type JobStatus = "uploading" | "queued" | "running" | "done" | "failed" | "canceled";

export interface JobOptions {
  start_s: number;
  seconds: number;
  quality: "fast" | "balanced" | "detailed";
  render: boolean;
  team_a: string;
  team_b: string;
  season: string;
}

export interface Card {
  track_id: number;
  name: string;
  team: string;
  grade: string;
  score: number;
  strengths: string[];
  weaknesses: string[];
  minutes: number;
  review_clips: number;
}

export interface JobResult {
  version: number;
  video: { fps: number; duration_s: number; width: number; height: number };
  range: { start_s: number; end_s: number };
  calibration: {
    mode: string;
    auto: { frame: number; seconds: number; markings: number; lines: number; fit_error_ft: number } | null;
  };
  tiles: {
    skaters: number;
    named: number;
    numbers_read: number;
    top_speed_mph: number | null;
    fastest: string | null;
    shots: number;
    passes: number;
    turnovers: number;
  };
  stars: Card[];
  cards: Card[];
  teams: Record<string, Record<string, number | null>>;
  events: { t: number; type: string; player: string; to: string | null; team: string; speed_mph: number | null }[];
  camera_shots: { start_s: number; end_s: number; status: string }[];
  files: Record<string, string>;
}

export interface Job {
  id: string;
  created_at: string;
  updated_at: string;
  status: JobStatus;
  filename: string;
  size_bytes: number | null;
  options: Partial<JobOptions>;
  progress: number;
  message: string | null;
  result: JobResult | null;
  error: string | null;
  started_at: string | null;
  finished_at: string | null;
  /** Short-lived download links, present once the job is done. */
  urls?: Partial<Record<"video" | "report" | "stats", string>>;
}

export type JobSummary = Pick<Job, "id" | "filename" | "status" | "progress" | "message" | "created_at" | "finished_at">;
