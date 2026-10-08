### The pipeline, one step at a time

**1 · Import models from Hugging Face.**
Nothing is trained from scratch. `rinkiq/models.py` downloads small YOLO detectors from
[`AlexRHaigh/Hockey-Vision`](https://huggingface.co/AlexRHaigh/Hockey-Vision), which were trained on NHL broadcast
video: players with their team, the puck, painted rink lines and faceoff dots. It also loads Ultralytics'
`yolov8n-pose` for body keypoints. The files are cached after the first download.

**2 · Detect + track.**
Every frame goes through the player model. ByteTrack links the detections over time so each skater keeps one
ID, and each track takes the team label it was given most often. A separate model looks for the puck.

**3 · Undo the camera.**
Broadcast cameras pan and zoom. ORB feature matching (with players and burned-in score graphics masked out)
estimates how the camera moved between frames. Every faceoff dot the dots model spots then pulls the
estimate back onto the real rink (*drift correction*).

**4 · Map to the rink.**
Four or more landmarks (dots, line/boards intersections) give a homography: a 3×3 matrix that sends any
pixel on the ice to rink feet. Each player's skates (the bottom-centre of the box) go through it. With no
calibration, the box height (≈ 5.5 ft) serves as a ruler, which is enough for speed.

**5 · Body orientation.**
On each player crop, the pose model finds the shoulders, hips and nose. Shoulder order tells front from
back, shoulder width tells how square to the camera the player is, and the nose breaks left/right. This
gives the yellow facing arrow and the *skating backwards %* stat. It's a heuristic, so treat it as experimental.

**6 · Stats** (`rinkiq/metrics.py`, plain pandas):

| Stat | How |
|---|---|
| Speed, top speed | drop glitches → Savitzky–Golay smoothing → derivative; top = fastest 0.5 s average |
| Distance, bursts, >15 mph % | from the smoothed path; burst = ≥ 0.3 s above 8 ft/s² acceleration |
| Possession | nearest skater within 6 ft of the puck |
| Pass / turnover / shot | possession changes; a shot is a > 35 mph release heading at the net |
| Goal-side % | while defending, closer to own net than the nearest attacker |
| In position % | goal-side, and within 25 ft of the player's usual spot in the team shape |
| Gap / space | distance to the nearest opponent while defending / attacking |
| Team shape | width, depth and hull area of the five skaters, every frame |
| Skate Score | percentile blend: top speed 25 · avg speed 15 · bursts 20 · in-position 20 · space 20 |

**7 · Present.** This app, plus a CLI (`python scripts/analyze.py film.mp4`) that writes CSVs.

### Honest limitations
* The puck is tiny and often hidden, so shot speed only exists when the puck is visible for several frames.
* Calibration is most accurate near the landmarks you click. Dot snapping fixes most of the drift during pans.
* Team labels come from jersey colour and can flip on unusual uniforms.
* "Out of position" is a simple, explainable rule, not a coach's judgement. Use it to find clips worth watching.
