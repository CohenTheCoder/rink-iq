# 🎓 Learn computer vision with RinkIQ

**Who this is for:** you know some Python but have never done computer vision. Hockey video is a great
teacher, because it has *everything*: objects to detect, players to follow over time, a camera that moves,
a flat rink to map, and noisy measurements to clean up. By the end you'll be able to explain how a program
turns TV footage into *"#3 hit 21.4 mph and was out of position for 6 seconds"*, and how we know those
numbers are right.

**Three ways to learn, same 7 steps:**
1. 📖 **Read** this guide.
2. 🧪 **Run** [`notebooks/learn_cv_step_by_step.ipynb`](../notebooks/learn_cv_step_by_step.ipynb). It shows
   the frame at every stage.
3. 🎛️ **Play** with the **🎓 Learn CV** tab in the app (`streamlit run app.py`), the same steps with sliders.

```
video ─► ① pixels ─► ② detect players (YOLO) ─► ③ track them (IoU/ByteTrack) ─► ④ undo the camera (ORB + RANSAC)
      ─► ⑤ pixels → rink feet (homography) ─► ⑥ speed (smoothing) ─► ⑦ body pose ─► stats ─► ✅ validate vs. ground truth
```

<details>
<summary>⚙️ Setup (5 minutes)</summary>

```bash
git clone https://github.com/CohenTheCoder/rink-iq.git && cd rink-iq
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyter matplotlib
jupyter notebook notebooks/learn_cv_step_by_step.ipynb
```
The first run downloads small YOLO models and a 10-second NHL sample clip from
[`AlexRHaigh/Hockey-Vision`](https://huggingface.co/AlexRHaigh/Hockey-Vision) on Hugging Face.
</details>

---

## ① A video is a stack of images, and an image is a grid of numbers
🧠 A video is just frames shown quickly, here 60 per second. Each frame is a NumPy array of shape
`(height, width, 3)`: **720 rows × 1280 columns × 3 colour values**, each 0–255. White ice is around 200+ and a
dark jersey around 30–80. *All of computer vision is math on this grid.*

```python
from rinkiq import models, pipeline
clip = models.download_sample_clip("Broadcast clip · neutral zone (MTL vs SJ)")
frame = pipeline.read_frame(str(clip), 0)      # OpenCV → BGR order!
frame.shape                                    # (720, 1280, 3)
```
⚠️ **Gotcha:** OpenCV stores colour as **B, G, R**. Convert with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`
before showing it with matplotlib, or the red team turns blue.

⚠️ **Gotcha:** arrays are indexed `frame[row, col]` = `frame[y, x]`, with **y growing downward** from the
top-left corner.

## ② Object detection: boxes, classes and confidence (YOLO)
🧠 A detector returns, for each object it recognises, a **bounding box** `(x1, y1, x2, y2)`, a **class**
(team-A player, team-B player, referee, goalie, puck…) and a **confidence** from 0 to 1. **YOLO** ("You Only
Look Once") is a convolutional network that looks at the whole image in **one pass** and predicts every box
at once, fast enough for video. RinkIQ downloads YOLO models that someone already **trained on NHL
broadcasts** and shared on Hugging Face (*transfer learning*: we didn't train anything).

🔍 [`rinkiq/models.py`](../rinkiq/models.py) → `models.load("player")`, then `model(frame, conf=0.05)`.

💻 **Try it:** slide the confidence threshold. Too **low** gives false alarms (fans, reflections in the glass);
too **high** misses real players. RinkIQ uses **0.3**. This is the **precision vs. recall** trade-off you'll
meet in every ML project.

✅ <details><summary><b>Check yourself:</b> why is the puck the hardest object?</summary>
It's ~5 pixels wide on a broadcast frame, blurs when shot, and hides behind sticks and skates. That's why
RinkIQ uses a separate puck model and only reports a shot speed when the puck is visible for several frames.</details>

## ③ Tracking: is this the same player as a moment ago?
🧠 Detection works on single frames. **Tracking** links detections over time so each skater keeps one ID.
The core measure is **IoU (intersection over union)**: the overlap of two boxes ÷ their combined area, from
0 (no overlap) to 1 (identical). At 60 fps, a player barely moves between frames, so their boxes overlap
a lot.

🔍 RinkIQ uses **ByteTrack**, which adds a **motion prediction** (where should this player be next?) and a
second matching pass for *low-confidence* boxes, so a partly hidden player isn't dropped. Each track keeps
the team label it was given most often (**majority vote**), which smooths out one-frame mistakes.

💻 Compare frames 0.1 s apart (most IoUs > 0.5) with 0.5 s apart (fast skaters have left their old box).
That's why trackers predict motion instead of relying on overlap alone.

## ④ The camera moves too: ORB + RANSAC
🧠 TV cameras pan and zoom. If we don't remove that, a standing player looks like they're skating. We find
**keypoints**, distinctive corners like ad logos and glass edges, using **ORB**, and match them between
frames. If most matches shift the same way, *that* is the camera's motion.

🧠 Some matches are wrong (they land on moving players). **RANSAC** handles this by repeatedly guessing a
motion from a few random matches and keeping the guess *most* matches agree with. Those agreeing matches are
the **inliers**, and the outliers are ignored. You'll see RANSAC everywhere robust fitting is needed.

🔍 [`rinkiq/camera.py`](../rinkiq/camera.py) masks out players and burned-in score graphics (which never
move with the camera) before matching. Every faceoff dot the dots model sees pulls the estimate back onto
the real rink (*drift correction*). Without that, drift reached ~40 ft during a full pan; with it, ~1–5 ft.

## ⑤ From pixels to rink feet: the homography
🧠 The ice is **flat**, and for any flat surface one **3×3 matrix**, a **homography**, maps camera pixels
to top-down coordinates. Four known points are enough to solve for it, like faceoff dots or where lines meet the
boards, whose real positions (in feet) come from the NHL rink spec in
[`rinkiq/rink.py`](../rinkiq/rink.py). Each player's position is the **bottom-centre of their box**, their
skates, which actually touch the ice.

```python
from rinkiq.rink import Calibration, pixels_to_rink
cal = Calibration.from_pairs([((332, 262), "Neutral dot · left · far"), ...])   # 4+ pixel ↔ landmark pairs
pixels_to_rink(np.array([[640, 600]]), cal.H)    # → [[x_ft, y_ft]] on a 200 × 85 ft rink
```
💻 **Try it:** nudge one clicked point by a few pixels. The error near that point stays small, but it
**grows far away from it**. Small errors are amplified with distance, which is exactly why RinkIQ keeps
re-anchoring on faceoff dots.

## ⑤b Automatic calibration: finding the rink with no clicking
🧠 Clicking landmarks is fine for one game but useless for a website where anyone uploads a video. So RinkIQ
solves it as a **search**:
1. Find everything painted: faceoff dots (inside a circle → end zone, else neutral zone), the center circle,
   goal frames, and the **blue / red lines**, fitted as straight lines with a **Hough transform**.
2. Each marking could be one of a few real positions. Try **every** combination that obeys common sense: left
   stays left, higher up is further away, nothing on both sides of the center line.
3. Fit a homography to *points and lines together*. A line gives 2 equations just like a point.
4. **Judge** each candidate on evidence it *wasn't* fitted to: painted-line boxes, faceoff circles, and whether the
   projected rink lands on **white ice** in the picture.

⚠️ **Three lessons from real film:**
* *Exactly enough constraints always fit perfectly.* 3 points + 1 line = 8 equations = 8 unknowns, so the "error" is 0
  even for a wrong match. Demand a spare constraint before believing a fit.
* *Symmetry fools you.* A rink looks the same mirrored or turned 180°. A real camera can't film a mirror image
  (the mapping keeps **handedness**, so its Jacobian determinant has a fixed sign), and it films from the near side.
  Encoding those physical facts as checks removed every flipped solution.
* *Check the ground truth itself.* My rink spec had the neutral-zone dots 10 ft out of place. Every downstream
  number looked plausible until a fit refused to agree with the paint.

🔍 [`rinkiq/autocal.py`](../rinkiq/autocal.py) · `draw_rink_overlay()` in [`rinkiq/rink.py`](../rinkiq/rink.py)
draws the fitted rink back onto the frame, the fastest way to *see* whether a calibration is right.

## ⑥ From positions to speed: smoothing noisy data
🧠 Speed = distance ÷ time. But boxes **jitter** by a few pixels every frame, and at 30 fps a 0.3 ft wobble
already looks like 9 ft/s of fake speed. So we **smooth** the path first: a **Savitzky–Golay filter**
fits a small polynomial through a sliding window of frames, which keeps real accelerations better than a
plain moving average.

🔍 [`rinkiq/metrics.py`](../rinkiq/metrics.py): drop impossible jumps (> 40 mph) → smooth → take the
derivative. *Top speed* is the fastest **0.5 s average**, not a single frame, so one glitch can't create a
record.

💻 **Try it:** raise the jitter and the raw speed goes wild. Widen the window and it calms down, but too wide
and real stops and sprints get flattened. Choosing that window is the **bias–variance trade-off**.

## ⑦ Body pose: which way is a player facing?
🧠 **Pose estimation** finds 17 body **keypoints** (nose, shoulders, elbows, hips, knees…), each with a
confidence. From the shoulders and hips we can tell which way a player faces: if the *right* shoulder
appears on the *left* of the image, they're facing the camera.

🔍 [`rinkiq/pose.py`](../rinkiq/pose.py) runs Ultralytics' `yolov8n-pose` on each player crop. Broadcast
players are only 60–100 px tall, so keypoints are rough. RinkIQ uses facing only for one coarse stat,
*skating backwards %*, and labels it experimental. **Knowing a signal's limits is part of the job.**

## ⑧ Who is who? Jersey numbers → names
🧠 A tracker gives *IDs* (#17, #42…), not *people*. A second small detector finds **digits** on each player
crop (upscaled 4×). Two digits side by side make a two-digit number. One frame can misread "63" as "3"
when an arm covers the 6, so each track **votes over time**, and a number is only confirmed once it wins
clearly. Team + number → name comes from a **roster**: the public NHL API, or one you paste in.

🔍 [`rinkiq/identity.py`](../rinkiq/identity.py). Bonus: if a player's track broke in two (hidden behind
someone, or skated out of frame), both pieces share team + number, so they're **merged back into one
player**. That's a simple form of *re-identification*.

## ⑨ Long videos: camera cuts and finding your place again
🧠 Broadcasts **cut** between cameras. A cut breaks tracking *and* the rink calibration, so RinkIQ detects
cuts by comparing colour **histograms** of consecutive frames (a cut makes them suddenly different) and starts
a new *shot*. To place the new view on the rink without clicking points again, it matches the frame's ORB
keypoints against a bank of **keyframes**, views it has already calibrated. Close-ups and replays that match
nothing are skipped and listed. Results are written to disk in chunks, so a full game can run overnight and
**resume** if it stops.

🔍 `run_video(..., job_dir="outputs/game1")` in [`rinkiq/pipeline.py`](../rinkiq/pipeline.py).

✅ <details><summary><b>Check yourself:</b> why re-check keyframes *during* a shot, not only after a cut?</summary>
Camera tracking drifts a little every frame. Matching against a known view resets the error to near zero.
It's the same reason a GPS app snaps you back onto the road.</details>

## 📊 From tracks to hockey stats (plain pandas)
Once players are in rink feet over time, everything else is data science, with no more vision needed.
Possession = nearest skater within 6 ft of the puck · pass/turnover = possession changes · shot = a release
over 35 mph toward the net · goal-side % · gap · team shape (convex hull area). See the table in
[`docs/HOW_IT_WORKS.md`](HOW_IT_WORKS.md).

**Who played well, and why?** [`rinkiq/report.py`](../rinkiq/report.py) turns every stat into a **percentile**
among the skaters in the video, then writes the strongest and weakest as sentences ("Top speed 24.1 mph,
#1 of 18"), blends them into a letter grade, and picks **three stars**. Because the reasons *are* the inputs,
there's nothing hidden. That matters whenever a model's output affects people.

**Drawing it back on the video** ([`rinkiq/render.py`](../rinkiq/render.py)): to keep a player's trail glued
to the ice while the camera pans, each past position is mapped to the shot's reference view with *that*
frame's camera matrix, then back into today's frame with the inverse of the current one.

## ✅ How do we know it's right? Validation against ground truth
🧠 Real game film doesn't come with true speeds, so we **simulate a game where we know the truth**
([`rinkiq/simulate.py`](../rinkiq/simulate.py)): scripted plays, an 88 mph one-timer, realistic tracking
noise, 15 % of puck detections dropped, and one winger who cheats up ice on purpose. Then we check whether
the pipeline recovers what we put in:

| Check | Result |
|---|---|
| Passes found | **25 / 25** |
| Shots found | **10 / 10**, speed error ≤ 3.3 % |
| Top speed, mean abs. error | **0.7 mph** |
| Cheating winger flagged "out of position" | **5 / 5** |

⚠️ Simulation tests the **math** (tracking cleanup, smoothing, stats). The **vision** side is checked
separately on a real NHL clip (calibration mean error 0.19 ft). Be clear about which part each test covers.

---

## 📖 Glossary
| Term | Meaning |
|---|---|
| **Frame / fps** | one image of the video / frames per second |
| **Bounding box** | `(x1, y1, x2, y2)` rectangle around an object |
| **Confidence threshold** | minimum score to keep a detection: precision ⇄ recall knob |
| **IoU** | overlap of two boxes ÷ their union (0–1) |
| **Tracking / track ID** | linking detections across frames / the identity that persists |
| **Keypoint (features)** | a distinctive, re-findable corner in an image (ORB finds them) |
| **Keypoint (pose)** | a body joint location (nose, shoulder, hip…) |
| **RANSAC** | robust fitting that ignores outliers by voting |
| **Homography** | 3×3 matrix mapping one plane (the ice in the image) to another (the rink diagram) |
| **Savitzky–Golay** | smoothing by fitting small polynomials in a sliding window |
| **Ground truth** | the known correct answer you compare against |
| **Re-identification** | recognising the same person again after they disappear (here: by jersey number) |
| **Keyframe** | a stored, already-calibrated view used to re-locate the camera after a cut |
| **Hough transform** | finds straight lines in an image by letting edge pixels vote for line positions |
| **Handedness** | whether a mapping keeps or flips orientation: a camera keeps it, a mirror flips it |

## 🧭 Where to go next
* **OpenCV-Python tutorials**: feature matching, homography and RANSAC, with pictures.
* **Ultralytics docs**: detection, tracking and pose with YOLO.
* **Hugging Face Computer Vision Course**: the deep-learning side, hands-on.
* **Exercise:** in `rinkiq/metrics.py`, change `_sustained_top_speed(g, seconds=0.5)` to `seconds=0.03`
  (about one frame), re-run `pytest`, and compare top speeds with the simulator's truth. That's the case for
  smoothing, in numbers.
