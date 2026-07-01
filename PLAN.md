# BlazePose → CoreML: Implementation Plan

Goal: an extremely optimized on-device pose detector for iOS, starting by porting
Google MediaPipe's **BlazePose** to CoreML and running it on the Apple Neural Engine (ANE).

> ## ⚡ Overriding priority: performance & speed
> This project's #1 goal is **inference latency** (and sustained throughput / thermals) on-device,
> ahead of accuracy, code simplicity, or feature breadth. Every design decision below should be made
> through that lens:
> - **ANE-resident** execution is mandatory, not nice-to-have — FP16, ML Program, ops chosen/structured
>   to avoid GPU/CPU fallbacks.
> - **Zero-copy** data path: `CVPixelBuffer` → model → outputs, with normalization baked into the graph;
>   no per-pixel CPU work, no needless `MLMultiArray` copies.
> - **Do less work**: detector-skip tracking, smallest viable variant, smallest viable input resolution,
>   and quantization/palettization whenever they hold accuracy.
> - **Measure everything**: latency (p50/p99), not just averages; per-layer compute-unit placement;
>   energy. A change isn't "done" until it's benchmarked on real hardware.
> - When in doubt, prefer the faster option and document the accuracy cost.

---

## 1. Background: what BlazePose actually is

BlazePose is **not one model**. It's a two-stage pipeline:

1. **Pose Detector** (`pose_detection.tflite`) — an SSD-style detector that finds a person
   and returns a coarse ROI (bounding box + a couple of keypoints used to compute rotation/scale).
   Runs on the full frame.
2. **Pose Landmark** (`pose_landmark_{lite,full,heavy}.tflite`) — takes a 256×256 crop aligned
   to the ROI and regresses **33 landmarks** with `(x, y, z, visibility, presence)` plus a
   segmentation mask. This is the model you run every frame.

Critically, a lot of the MediaPipe pipeline lives **outside** the neural nets, in C++ "calculators":
anchor generation + box decoding + **non-max suppression** for the detector, and ROI cropping /
rotation / coordinate un-projection around the landmark model. **CoreML will only contain the
neural net.** We must reimplement the pre/post-processing ourselves (Swift, or partly as extra
graph ops). Plan for that explicitly — it's where most bugs hide.

Variants and tradeoffs (pick per target device/use-case):
- `lite` — fastest, lowest accuracy
- `full` — balanced (good default)
- `heavy` — most accurate, heaviest

Start with **full**; benchmark `lite` and `heavy` once the pipeline works.

---

## 2. Conversion strategy (decision)

**Chosen path: TFLite → PyTorch reimplementation → CoreML (via coremltools Unified API).**

Why not the alternatives:
- **Direct TFLite → CoreML**: coremltools has *no* TFLite frontend. Dead end.
- **TFLite → ONNX → CoreML**: possible (tf2onnx / `onnx2tf` / PINTO's `tflite2tensorflow`), but
  BlazePose uses custom TFLite ops (e.g. transpose-conv with bias) that translate messily, and
  the `onnx-coreml` frontend is deprecated. Fragile.
- **PyTorch route**: coremltools has first-class PyTorch support (`torch.jit.trace` / `torch.export`),
  and **someone already did the hard part** — [zmurez/MediaPipePyTorch](https://github.com/zmurez/MediaPipePyTorch)
  reimplements BlazePose (and BlazeFace/BlazePalm) in PyTorch *and* ships scripts that load the
  weights straight from the official `.tflite` files (BatchNorm already folded into conv weights by TFLite).

So: get a numerically-correct PyTorch model first, verify it matches TFLite, *then* convert. CoreML
conversion is then the easy, well-trodden step.

Reference assets to pull from:
- `zmurez/MediaPipePyTorch` — PyTorch model defs + weight-loading from tflite.
- `geaxgx/depthai_blazepose` — bundles the actual `.tflite` files (Full/Lite/Heavy) and clean
  Python reference pre/post-processing to copy.
- `qualcomm/MediaPipe-Pose-Estimation` (HuggingFace) — sanity reference / alt weights.
- `vidursatija/BlazeFace-CoreML` — example of this exact PyTorch→CoreML pattern end-to-end.

---

## 3. Project layout

```
PoseML/
  PLAN.md                     # this file
  models/
    tflite/                   # original mediapipe .tflite (source of truth for weights)
    pytorch/                  # ported nn.Module + state_dicts
    coreml/                   # exported .mlpackage outputs
  python/
    convert/                  # weight loading + coremltools conversion scripts
    verify/                   # numerical parity tests (tflite vs torch vs coreml)
    anchors/                  # anchor generation + reference decode/NMS in numpy
  ios/
    PoseML/                   # Swift package / app: capture → preprocess → CoreML → decode → overlay
  Makefile                    # reproduce the whole conversion from scratch
```

---

## 4. Phased implementation

### Phase 0 — Environment & assets (½ day)
- Python env (3.11) with pinned `coremltools` (latest 8.x), `torch`, `tensorflow`/`tflite-runtime`
  (only to run the reference model), `numpy`, `opencv-python`, `netron`.
- Download the official `.tflite` models (detector + the three landmark variants) into `models/tflite/`.
- Open each in **Netron**; record exact input/output tensor names, shapes, dtypes, and the op list.
  Note any custom ops. This document is the contract for everything downstream.
- **Exit criteria:** can run each `.tflite` in Python and get outputs on a test image.

#### Phase 0 result — discovered I/O contract (captured in `models/tflite/io_contract.json`)

**Detector** `pose_detection.tflite` (442 tensors):
- in:  `input_1` `[1, 224, 224, 3]` f32 (NHWC)
- out: `Identity` `[1, 2254, 12]` (per-anchor box+keypoint regressions), `Identity_1` `[1, 2254, 1]` (scores)
- → **2254 anchors**; decode + NMS happen outside the net (Phase 3).

**Landmark** `pose_landmark_{lite,full,heavy}.tflite` (identical signatures; 492 / 535 / 1098 tensors):
- in:  `input_1` `[1, 256, 256, 3]` f32 (NHWC)
- out:
  - `Identity`   `[1, 195]`           → **39 landmarks × 5** (x, y, z, visibility, presence) in crop space
  - `Identity_1` `[1, 1]`             → overall pose-presence score
  - `Identity_2` `[1, 256, 256, 1]`   → segmentation mask
  - `Identity_3` `[1, 64, 64, 39]`    → 39-channel heatmap
  - `Identity_4` `[1, 117]`           → **39 world landmarks × 3** (metric x, y, z)
- Note: 39 = 33 body + 6 auxiliary points. For a latency-first build we likely only need
  `Identity` (+ maybe `Identity_4`); **prune `Identity_2`/`Identity_3` (mask + heatmap) from the
  CoreML graph** unless a feature needs them — they're a large chunk of compute/output bandwidth.

### Phase 1 — PyTorch parity ✅ DONE
Approach changed after Phase 0: **every op is standard** (CONV_2D, DEPTHWISE_CONV_2D, ADD, PAD,
RESHAPE, RESIZE_BILINEAR, MAX_POOL_2D, CONCATENATION, DEPTH_TO_SPACE, LOGISTIC — no custom ops).
Rather than hand-adapt `zmurez/MediaPipePyTorch` (which targets an older BlazePose version), and
rather than tflite→ONNX→torch (which would pull `tensorflow`/`onnx` and force a protobuf downgrade
that breaks coremltools' `protobuf 7.35`), we built a **small generic tflite→PyTorch graph
converter** (`poseml.tflite_port`):
- `parser.py` — reads graph structure + op options from the flatbuffer (pure-python `tflite`
  schema pkg) and resolved fp16→f32 weights from the LiteRT interpreter. Folds `DENSIFY` (sparse
  detector weights) to dense via a zero-input reference invoke.
- `torch_graph.py` — `TfliteModule`, a traceable `nn.Module` that executes the graph. Tensors kept
  **NHWC-canonical** (matches TFLite axis semantics); conv/pool transpose to NCHW per-op (coremltools
  folds these later). `build(path)` returns the module.

Bugs found & fixed via intermediate-tensor diffing:
- **`DEPTH_TO_SPACE`**: TFLite uses **DCR ordering** (channel innermost) ≠ `F.pixel_shuffle` (CRD).
  Reimplemented with explicit reshape/permute. (Was the detector's only structural break.)
- Resize: TFLite `half_pixel_centers=True` → torch `align_corners=False` (verified to 1e-7 isolated).
- Explicit-`PAD` + `VALID` stride-2 convs (not `SAME`) — so the classic padding-drift trap didn't apply.

**Parity gate** (`poseml.verify.parity`, `make verify`): because fp16 weights make bit-exact
impossible and the seg-mask is numerically ill-conditioned (TFLite's *own* XNNPACK vs reference
kernels disagree by ~0.38 on it), we gate each output at `err(torch vs ref) ≤ max(5e-3, 3×|xnnpack−ref|)`
— "match TFLite at least as well as TFLite matches itself."

**Result — all 4 models PASS** (pytest `python/tests/test_parity.py`):
- landmark `Identity` (33 pose coords): ~1–10e-3 (well-conditioned outputs ≤5e-3)
- detector `Identity`/`Identity_1`: ~6e-4
- seg-mask passes via the noise-floor rule (still slated for pruning in Phase 2)

### Phase 2 — CoreML conversion (1–2 days)
- Trace each model (`torch.jit.trace` with a fixed 256×256 / detector-res dummy input).
- Convert with coremltools Unified API:
  - `convert_to="mlprogram"` (ML Program, not the legacy NeuralNetwork format)
  - `compute_precision=ct.precision.FLOAT16` (ANE is FP16-only)
  - `compute_units=ct.ComputeUnit.ALL`
  - **Image input**: use `ct.ImageType` so the model accepts a `CVPixelBuffer` directly
    (zero-copy from the camera). Bake the BlazePose normalization (scale `1/255`, and any
    mean/std) into `scale`/`bias` so Swift does no per-pixel math.
  - Set minimum deployment target (e.g. iOS 17) to unlock newer ops/optimizations.
  - Name the outputs explicitly for clean Swift access.
- Re-run parity: CoreML (on macOS, via `coremltools` predict / `MLModel`) vs PyTorch vs TFLite.
- **Exit criteria:** `.mlpackage` for detector + landmark(full), parity within FP16 tolerance.

### Phase 3 — Post/pre-processing & glue (3–4 days)
This is the real work; the nets are the easy part.
- **Anchors**: generate the detector's SSD anchor grid (match MediaPipe's
  `SsdAnchorsCalculatorOptions` exactly — strides, aspect ratios, counts). Precompute once,
  ship as a constant array.
- **Detector decode**: sigmoid scores → threshold → decode boxes/keypoints against anchors → **NMS**.
  Implement in Swift (or Accelerate/`vDSP`). Validate against the numpy reference from Phase 2.
- **ROI computation**: from detector keypoints, compute center, rotation, and scale; build the
  affine crop for the landmark model (256×256, rotated). Match MediaPipe's `rect_transformation`.
- **Landmark decode**: map normalized landmark coords back through the inverse affine into original
  image space; apply `visibility`/`presence` sigmoids.
- **Tracking optimization**: like MediaPipe, run the *detector only when needed* — once a pose is
  found, derive next-frame ROI from the previous landmarks and skip the detector. Huge speedup.
- **Exit criteria:** end-to-end Python pipeline produces correct 33-landmark overlays on test images/video.

### Phase 4 — iOS integration (3–5 days)
- Swift package + minimal app: `AVCaptureSession` → `CVPixelBuffer` → CoreML (Vision or direct
  `MLModel` with `MLFeatureProvider`).
- Port the Phase-3 decode/NMS/ROI logic to Swift (reuse the numpy reference as the spec).
- Use **Vision** (`VNCoreMLRequest`) for convenient pixel-buffer handling + auto orientation,
  *or* drive `MLModel` directly for maximum control over the two-stage + tracking loop. Recommend
  direct `MLModel` here because the two-model tracking pipeline doesn't fit Vision's one-request model cleanly.
- Render landmarks over the camera preview (Metal or SwiftUI Canvas overlay).
- **Exit criteria:** live pose overlay on-device.

### Phase 5 — Optimization & benchmarking (ongoing)
- **Measure first**: use Xcode's Core ML Instruments template + `coremltools` performance report
  to see per-layer compute-unit placement. Confirm the landmark model lands on **ANE**, not GPU/CPU.
- Fix ANE fallbacks: some ops force GPU/CPU; reshape/restructure or use coremltools passes to keep
  the graph ANE-resident.
- **Quantization / palettization**: try `coremltools.optimize` (FP16 → INT8 / 6-bit palettization,
  weight pruning) and measure the accuracy↔latency↔size tradeoff per variant.
- Pipeline the two models, use the tracking shortcut (Phase 3), and minimize CPU↔ANE copies
  (keep everything in `CVPixelBuffer`/`MLMultiArray` form).
- Target: full model comfortably real-time (well under 16 ms/frame for 60fps) on recent A-series.
- **Exit criteria:** documented latency/accuracy table for lite/full/heavy on target devices.

---

## 5. Top risks & mitigations
- **Pre/post-processing mismatch** (anchors, NMS, ROI affine, coord un-projection) — the #1 source
  of "model runs but output is garbage." Mitigation: keep a numpy reference and diff Swift against it
  at every stage, not just end-to-end.
- **Conv padding drift** (TFLite SAME vs PyTorch) — caught in Phase 1 parity tests.
- **ANE op fallbacks** silently moving work to GPU/CPU — caught with Instruments in Phase 5.
- **FP16 accuracy loss** — usually fine for pose, but verify visibility thresholds still behave.
- **Licensing**: MediaPipe models are Apache-2.0; fine to use. Note attribution.

## 6. Suggested first concrete step
Phase 0 + the start of Phase 1: stand up the Python env, pull the `.tflite` files and the
`zmurez/MediaPipePyTorch` port, and get a single test image producing matching outputs from both
TFLite and PyTorch. Everything else builds on that parity foundation.
