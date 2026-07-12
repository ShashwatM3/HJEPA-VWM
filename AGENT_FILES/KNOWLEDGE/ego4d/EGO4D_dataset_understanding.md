# EGO4D dataset understanding

> Purpose: a fully-accurate reference on what EGO4D actually is — the nature of the videos,
> how the released data is structured on disk, its frame rate and resolution, its licensing
> and download mechanics — so that data-pipeline work (`data.py`, `config.py`, a future
> chunker/downloader) can be built against real specifications instead of assumptions.
> Sourced from the official EGO4D documentation (ego4d-data.org), the CVPR 2022 paper
> (Grauman et al., "Ego4D: Around the World in 3,000 Hours of Egocentric Video," arXiv:2110.07058),
> the `facebookresearch/Ego4d` CLI repository, and this project's own
> [`EGO4D_DATASET_CHAT_CONTEXT.md`](EGO4D_DATASET_CHAT_CONTEXT.md) handoff note. Companion
> document: [`EGO4D usage options.md`](EGO4D%20usage%20options.md).

## 1. What EGO4D is

EGO4D (Egocentric 4D Perception) is a massive-scale **egocentric** (first-person, head-mounted
camera) video dataset released by a consortium of 88 researchers across 13 universities and
labs, led by Meta AI / FAIR, and published at CVPR 2022. It contains **3,670 hours** of
daily-life activity video, captured by **931 unique camera wearers** across **74 worldwide
locations** in **9 countries** (the count of camera wearers and locations has been reported
consistently as 931 and 74 across the official paper, the project site, and third-party
citations; hour totals crept slightly over dataset versions — the original paper title rounds
to "3,000 hours," a later journal revision is titled "3,600 hours," and the dataset's current
released total is 3,670 hours).

The defining property is the **viewpoint**: every camera is worn on the head or body of the
person performing the activity, so the frame is whatever that person is looking at while they
cook, garden, do carpentry, play with children, shop, do laundry, take a walk, socialize, or
do dozens of other unscripted daily-life activities. This is categorically different from
third-person/exocentric footage (a static or externally-operated camera watching a subject).
Participants were recruited to record their own everyday routines rather than perform scripted
actions for a camera crew, so the camera continuously translates and rotates with the wearer's
head and body movement — walking, turning to look at something, reaching, bending down — for
the entire duration of each recording.

Beyond RGB video, subsets of EGO4D carry additional modalities: synchronized audio, 3D scans
of the environment (for videos recorded in a subset of "3D scan" locations), eye-gaze tracking,
stereo video, and in some sessions multiple synchronized egocentric cameras recording the same
event from different wearers. No single video carries every modality; each is annotated per
video in the metadata (see §4).

**Ethics and privacy.** Participants signed consent as part of data collection, and EGO4D
underwent an ethics review and de-identification process (e.g. redaction of bystander faces
and other personally identifying content in some intervals — recorded per-video as
`redacted_intervals` in the metadata). This is separate from license access controls (§6).

## 2. Video capture: devices, frame rate, resolution

### Capture devices

Recordings were made with **seven different head-mounted camera models**, deliberately varied
across sites so that no single Phase 1 vision model could overfit to one camera's optical or
compression signature: GoPro (various models), Vuzix Blade, Pupil Labs, ZShades, ORDRO EP6,
iVue Rincon 1080, and Weeview. Because these are heterogeneous consumer/prosumer devices, the
**raw, as-captured footage is not uniform** — different devices shoot at different native frame
rates, resolutions, and aspect ratios, and even a single device's raw export ("video
components," see §3) can vary in codec, audio sample rate (32kHz vs 48kHz), and channel count
(mono vs stereo) from clip to clip.

### Canonical (post-processed) frame rate and resolution

To make the dataset usable without every downstream consumer re-deriving per-device
normalization, EGO4D ships a **canonical** processed form of every video:

- **Frame rate:** canonical videos are standardized to a constant **30 FPS**.
- **Compression:** canonical full videos are VP9-encoded at CRF 41; canonical *clips* (shorter
  extracts tied to specific benchmark annotations) are VP9 at CRF 18 (higher quality, since
  clips are the unit models are usually trained/evaluated on). Audio is AAC.
- **Aspect ratio:** the sample aspect ratio is normalized to 1:1 (square pixels) across all
  canonical output, regardless of the source device's raw pixel aspect ratio.
- **Native/display resolution:** canonical full-scale video keeps the source recording's native
  display resolution (commonly reported around 1920×1080 or 1440×1080 depending on device;
  resolution is per-video metadata, not a single fixed constant across the whole corpus — see
  `display_resolution_width`/`display_resolution_height` in §4).
- **Downscaled release (`video_540ss` / `clips_540ss`):** EGO4D additionally publishes a
  spatially downscaled version of every full video and every clip, rescaled so the **shorter
  side is 540 pixels** ("540ss" = "540, shorter side"), with matching downscaled annotation
  coordinates (bounding boxes, gaze points, etc.) in a parallel `annotations_540ss` package.
  This is the practical, storage-manageable version most projects should start from: the
  full-resolution `full_scale` download is roughly **7 TB**, while `video_540ss` is roughly
  **5 TB**. Both are still at the canonical 30 FPS.

So the two axes normalize independently: **frame rate is always 30 FPS in the canonical
release** (regardless of device), while **spatial resolution has two official release tiers** —
native/display resolution (`full_scale`) or a 540-px-shorter-side downscale (`video_540ss`).
Neither canonical tier resamples to a small model-input size like 256×256; that crop/resize
step is left to the consumer's own data pipeline, exactly as this project's `data.py` already
does for SSv2 (`_resize_shorter_side` then `_crop` to 256×256).

## 3. How the released dataset is structured

EGO4D data exists in three layered forms, from rawest to most processed:

1. **Video components** — the raw, as-captured export closest to what the camera wrote to its
   SD card. These are ancillary/archival; properties (timebase, fps, audio format) are
   inconsistent across devices and are not the recommended training input.
2. **Canonical videos** — full-length videos, one per recording session, produced by
   concatenating and normalizing that session's video components into a single, consistently
   encoded file (30 FPS, 1:1 SAR, VP9/MP4 or WebM container). This is the unit identified by a
   `video_uid` and is what `full_scale` / `video_540ss` downloads contain.
3. **Canonical clips** — shorter segments trimmed out of a canonical video, each tied to a
   specific benchmark's annotations and identified by a `clip_uid`. Clip length varies by
   benchmark: Hand-and-Object (FHO) clips are 5-minute intervals with 8 seconds of padding
   before/after; Episodic Memory tasks (NLQ/VQ/MQ) use variable-length clips (typically
   6–10 minutes, capped at 20 minutes); Audio-Visual diarization (AV) clips are roughly
   5 minutes. Clip frame ranges are expressed as half-open intervals `[start_frame, end_frame)`
   against the parent canonical video.

Each canonical video is **long-form and untrimmed** relative to typical action-recognition
clips: EGO4D videos average roughly **24 minutes** and range up to **7 hours** in a single
recording session — this is the single largest structural difference from SSv2's few-second
clips (§8), and it is precisely the property this project's chat-context handoff
([`EGO4D_DATASET_CHAT_CONTEXT.md`](EGO4D_DATASET_CHAT_CONTEXT.md)) flags as requiring a new
chunking step before the data can feed the existing `SSV2Dataset`-style two-window loader.

### Video and clip metadata

Every video and clip carries a structured metadata record (documented at
`ego4d-data.org/docs/data/metadata/`). Key fields:

| Field | Meaning |
|---|---|
| `video_uid` | Primary unique identifier for a canonical video. |
| `origin_video_id` | The recording institution's own local ID for the session. |
| `video_source` | Which of the consortium's collecting institutions captured it. |
| `device` | Which of the 7 camera models recorded it. |
| `fb_participant_id` | Sequential anonymized participant number. |
| `fps`, `num_frames`, `duration_sec`, `mp4_duration_sec` | Per-video timing/frame-count facts (canonical fps is 30, but this field is authoritative per file). |
| `display_resolution_width/height`, `sample_resolution_width/height` | Output vs. capture pixel dimensions. |
| `video_codec`, `is_stereo` | Encoding and stereo-capture flags. |
| `scenarios` | Free-form activity-type labels (e.g. cooking, carpentry, gardening — hundreds of scenario tags exist across the corpus). |
| `physical_setting_name` | Named location, when a 3D scan exists for that setting. |
| `has_imu`, `has_gaze` | Whether IMU or eye-gaze streams accompany this video. |
| `redacted_intervals` | Time ranges where privacy redaction was applied. |
| `split_em`, `split_av`, `split_fho` | Per-benchmark train/val/test split assignment (splits are defined **per benchmark**, not once globally — see §5). |

Clips inherit a parallel structure (`clip_uid`, `video_start_frame`/`video_end_frame`, and a
`clip_metadata` object with clip-specific resolution/codec/duration), since a `clips_540ss`
clip's resolution differs from its parent `full_scale` video's resolution.

### Narrations

Independent of the benchmark annotations, a large fraction of EGO4D video is covered by dense,
free-text **narrations** — human annotators watched the footage and wrote short present-tense
descriptions of what the camera wearer is doing, timestamped roughly every few seconds (this is
the raw material several EGO4D-derived captioning/video-language corpora, e.g. narration-clip
pairings used by third-party papers, are built from). Across the corpus this amounts to on the
order of several million narration instances covering well over a thousand distinct verbs and
several thousand distinct nouns. Narrations are a separate annotation layer from the five
benchmark-specific annotation sets described next, and — like everything else in EGO4D — they
carry no discrete action-class label the way SSv2's 174 templates do (§8).

## 4. Benchmark tasks (context, not required for this project's use)

EGO4D ships annotations for five benchmark families, each with its own clip definition and
train/val/test split (`split_em`, `split_av`, `split_fho`, etc., independent per benchmark):

1. **Episodic Memory** — locate when/where something happened earlier in a long video, given a
   natural-language, visual, or object query (e.g. "where did I leave my keys?").
2. **Hand-Object Interactions (FHO)** — temporal localization and state-change classification of
   how the wearer's hands manipulate objects; this family also includes a "forecasting
   hand-object interaction" task (predicting the next action/object-state before it happens).
3. **Audio-Visual Diarization (AV)** — speaker localization, tracking, diarization, and speech
   transcription in egocentric recordings.
4. **Social Interactions** — group-dynamics understanding, including "Looking at Me" and
   "Talking to Me" classification.
5. **Forecasting** — predicting future motion/action beyond the current observation window
   (distinct from, but related in spirit to, FHO's interaction-forecasting task).

This project does not need any of these benchmark annotation packages — the training pipeline
here is self-supervised (predicts future frozen-encoder latents, not benchmark labels), so only
the **video** data tier (`full_scale` or `video_540ss`) is relevant, not `annotations`,
`3d`/`3d_scans`, `imu`, or any of the precomputed-feature packages. This mirrors how the
existing `SSV2Dataset` already globs `.webm` files with zero dependency on SSv2's
`labels.json`.

## 5. Splits

EGO4D benchmark tasks use fixed, **video-level** train/val/test splits (an entire canonical
video is assigned to exactly one split, so no video's content leaks across splits) at
approximately **70% / 15% / 15%**, with test-set annotations withheld server-side for blind
leaderboard evaluation. Because splits are stored per-benchmark (`split_fho`, `split_em`,
`split_av`, ...) rather than once globally, a video can in principle carry different split
membership across benchmarks — but for a benchmark-agnostic use (as here), any one benchmark's
split column, or a simple custom split by `video_uid`, is a reasonable and license-compliant
way to separate train/val video sets, exactly as SSv2's own train/validation directories do.

## 6. Access, licensing, and download mechanics

**License.** EGO4D is not open-download; access requires reviewing and signing the EGO4D
license agreement at `ego4d.dev/request/ego4d`, either as an individual or on behalf of an
institution. Approval has historically taken roughly **48 hours**, after which AWS S3 access
credentials arrive by email. Those credentials **expire after 14 days**, so the intended
workflow is to download the needed subset promptly rather than stream repeatedly from S3;
expired credentials can be renewed. The license requires citing the EGO4D paper in any resulting
publication. This document does not restate the license's exact permitted-use clauses (research
vs. commercial vs. redistribution) verbatim, since those are the kind of legal terms that must
be read from the signed agreement itself rather than paraphrased secondhand — verify directly
against the EULA text at `ego4d-data.org` before any use that depends on the exact scope of
permission.

**Download tool.** Meta publishes an official CLI (`pip install ego4d`, source at
`github.com/facebookresearch/Ego4d`) that reads the AWS credentials from
`~/.aws/credentials` and pulls named dataset packages, e.g.:

```bash
ego4d --output_directory="~/ego4d_data" --datasets full_scale annotations
ego4d --output_directory="~/ego4d_data" --datasets video_540ss annotations_540ss
```

Each requested package (`full_scale`, `video_540ss`, `clips`, `clips_540ss`, `annotations`,
`annotations_540ss`, `imu`, `3d`, `3d_scans`, precomputed feature packages, benchmark model
checkpoints, etc.) ships with its own `manifest.csv` enumerating contents and metadata, and the
CLI supports filtering by benchmark or by a supplied UID list so a project does not have to pull
the entire multi-terabyte corpus. This is the mechanism this project's chat-context handoff
refers to when it says to "download only that subset via the Ego4D CLI using `video_540ss` and
a UID file" — `video_540ss` is a real, first-class CLI dataset option, and UID-filtered partial
download is a supported, intended usage pattern, not a workaround.

**Practical size/time.** At typical broadband speeds (~100 Mbps), downloading the full ~7 TB
`full_scale` package would take on the order of a week; the smaller, still-30-FPS
`video_540ss` package (~5 TB) is proportionally faster, and a UID-filtered partial pull (the
plan for this project, targeting ~190–210 source hours rather than all 3,670) is smaller still.

## 7. What this means for a data pipeline built like this project's

Everything below is a direct, verified consequence of the specifications above, relevant to the
migration plan already sketched in
[`EGO4D_DATASET_CHAT_CONTEXT.md`](EGO4D_DATASET_CHAT_CONTEXT.md):

- **Frame rate is 30 FPS**, not SSv2's 12 FPS — confirmed as the canonical, dataset-wide
  constant (not a per-video variable) for both `full_scale` and `video_540ss`. This is the
  reason the chat-context note's fps-handling decision (Option A: re-encode chunks down to
  12 FPS during chunking so `config.py`'s `frame_stride=2`, `horizon_k=4` stay meaningful; or
  Option B: keep native 30 FPS and widen `frame_stride`/`horizon_k`) is a real, necessary step,
  not a hypothetical one.
- **Videos are long-form** (average ~24 minutes, up to 7 hours), not short clips — confirmed,
  and confirmed structurally different from SSv2's 2–6 second clips. A chunking step (splitting
  each long source video into short training-length segments, by source video and not by chunk,
  to avoid train/val leakage) is mandatory before the existing glob-based `SSV2Dataset` pattern
  (which assumes one directory of short, independently-sampleable clip files) can apply
  unchanged.
- **File format is MP4/WebM with VP9 video**, matching the codec family SSv2 already uses
  (SSv2 is also VP9-in-WebM), so the existing decord-based, `num_threads=1` decode path in
  `data.py` should be codec-compatible without new dependencies; only the file-extension glob
  (`*.webm` → also matching `*.mp4`, since EGO4D's canonical container is commonly `.mp4`) needs
  generalizing, exactly as the chat-context note specifies.
- **No label dependency exists or is needed** — EGO4D's benchmark annotations are irrelevant to
  this project's self-supervised objective, so the same zero-label, pure-glob loading pattern
  SSv2 already uses carries over unchanged in spirit.
- **`video_540ss` is the right download tier**, not `full_scale` — it is a real, officially
  supported, half-the-size, still-30-FPS package, and downscaling to a manageable spatial size
  before this project's own resize-to-256 step reduces both download volume and one-time
  chunking/decode cost without losing anything the 256×256 training resolution would have kept
  anyway.

## 8. Comparison with Something-Something V2 (SSv2)

SSv2 statistics below are drawn from this repository's own code (`data.py`, `config.py`,
`AGENT_FILES/AGENTS.md`) plus the dataset's public documentation (Goyal et al., "The
'something something' video database for learning and evaluating visual common sense,"
arXiv:1706.04261; the TwentyBN/Qualcomm dataset release).

| Property | Something-Something V2 (current) | EGO4D |
|---|---|---|
| Camera viewpoint | Third-person / exocentric: a static camera on a desk or tripod films a person's hands manipulating an object in front of it | First-person / egocentric: a head-mounted camera moves with the wearer through real daily-life activity |
| Total corpus size | 220,847 short clips (168,913 train / 24,777 validation / 27,157 test, test unlabeled) | 3,670 hours across roughly 9,600+ long-form recording sessions from 931 camera wearers |
| Typical clip/video length | ~2–6 seconds per clip (crowd-workers acted out a short templated action) | Long-form sessions, average ~24 minutes, up to 7 hours; not pre-segmented into action-length clips |
| Native frame rate | 12 FPS | 30 FPS (canonical) |
| Native/shipped resolution | 240p on the shorter side | Native display resolution (commonly ~1080p-class) for `full_scale`, or 540px shorter side for `video_540ss`; both far above SSv2's 240p |
| File format | `.webm`, VP9 video codec | `.mp4` (or `.webm`) canonical container, VP9 video codec (same codec family) |
| Labels | 174 fixed, template-based action classes (e.g. "Dropping [something] into [something]"), each clip's actual objects named by the crowd-worker who filmed it | No fixed action-class taxonomy; instead dense free-text narrations plus five separate benchmark annotation families (episodic memory, hand-object interaction, AV diarization, social interaction, forecasting), none of which this project consumes |
| This project's loader dependency on labels | None — `SSV2Dataset` globs `*.webm` with zero use of `labels.json` | None needed — same pattern applies |
| Motion / scene content per clip | A largely static background and camera; only a small foreground object or the actor's hands move; consecutive frames are visually very similar (the "95% similar frame" observation that motivated this migration) | The entire frame shifts every step from head/body motion, in addition to whatever hand-object or social activity is happening; scene content is far more diverse (74 locations, 9 countries, unscripted daily life vs. a single filming setup) |
| Access | Public dataset with a `labels.json`/split JSON download, no signed license required | Requires signing the EGO4D license agreement (~48 hour approval) and downloading via the official CLI against time-limited AWS credentials |
| Effective prediction-horizon consequence at this project's current config (`t_ctx=8`, `frame_stride=2`, `horizon_k=4`) | ≈0.33 seconds of real motion per training window at 12 FPS | The same frame-index math at native 30 FPS would compress to ≈0.13 seconds of real motion — meaningfully less — unless the fps or stride/horizon config is deliberately adjusted (see §7 and `EGO4D_DATASET_CHAT_CONTEXT.md` §"Migration specifics") |

The headline structural fact this comparison confirms is the one the chat-context handoff
already identified from direct inspection of the footage: SSv2's exocentric, static-camera,
small-object-manipulation setup produces near-identical consecutive frames, which is the direct
mechanical cause of the `coarse_vs_copy_ratio` failure this project's KANBAN history documents
across dozens of runs (copying the present latent forward is a strong baseline precisely because
present and future are almost the same embedding). EGO4D's egocentric, whole-frame-shifting,
multi-domain footage is structurally the opposite of that failure mode on every axis in the
table above, which is the dataset-level reason it is the migration candidate — independent of,
and prior to, any question of how to sequence the migration itself (covered in
[`EGO4D usage options.md`](EGO4D%20usage%20options.md)).
