# Basketball Action Recognition and Performance Analytics

An end-to-end pipeline that turns basketball broadcast video into structured
player statistics: player detection and tracking, ball trajectory estimation,
shot attempt detection, possession and team classification. It runs on a
single Google Colab notebook with a free GPU — no specialist hardware.

This was my final-year dissertation project at the University of Surrey. The
aim is not to compete with commercial systems like Second Spectrum on
accuracy. It is to show that the same kind of pipeline can be assembled from
open-source components and academic datasets, and evaluated against published
benchmarks.

## The pipeline

1. **Player detection** — YOLOv8n, fine-tuned on 12,384 frames from 15
   basketball sequences of the SportsMOT dataset.
2. **Tracking** — ByteTrack, evaluated with the motmetrics library.
3. **Ball trajectory** — a specialist ball detector was planned on
   DeepSportradar; its 3D annotations made that impractical within the
   timeline, so the pipeline falls back to COCO's pretrained `sports_ball`
   class and smooths the sparse detections with a 2D constant-velocity
   Kalman filter.
4. **Analytics** — a rule-based shot attempt detector (ball trajectory vs a
   defined hoop region), possession from ball–player proximity, and team
   classification by K-means on jersey colour.
5. **Evaluation** — detection and tracking metrics reported against the
   SportsMOT validation set, so the results can be compared with the
   published literature.

## Results, reported honestly

- Player detector: **0.971 mAP@50** on the SportsMOT validation set.
- Tracking: **MOTA 0.875**, **IDF1 0.555** across three validation sequences.
  The gap between the two is the interesting finding: detection is close to
  solved, but ByteTrack without an appearance embedding loses player identity
  during occlusions (~50 ID switches per sequence). A learned
  re-identification embedding is the clearest next step.
- Shot detection: limited by the fallback ball detector (balls found in ~13%
  of frames). The rule itself validated qualitatively; it needs a specialist
  ball detector to be quantitatively useful.

The full write-up — literature review, design rationale, evaluation and
failure cases — is in my dissertation; ask me for it.

## Running it

Open `basketball_analytics.ipynb` in Google Colab with a GPU runtime and run
top to bottom. The notebook is organised in sections:

- **A (Sections 1–4):** setup, utilities, video acquisition
- **B (Sections 5–6):** detection and tracking on your own clip with
  pretrained weights
- **C (Sections 7–13):** dataset download (SportsMOT via HuggingFace,
  DeepSportradar via Kaggle) and custom training — needs free API
  credentials, handled through Colab Secrets
- **D (Sections 14–18):** ball trajectory, shot attempts, possession, teams
- **E (Sections 19–22):** evaluation against SportsMOT ground truth, shot
  charts and the player dashboard

Datasets are downloaded from their official sources and are subject to their
own licences; none of their frames or annotations are redistributed here.

## Limitations

Single camera angle, offline processing, pixel-coordinate analytics (no court
calibration), and hoop regions defined manually per clip. Each of these, and
the path to removing it, is discussed in the dissertation's future work
chapter.
