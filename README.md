# PoseML

An extremely optimized on-device pose detector for iOS. First milestone: port MediaPipe
**BlazePose** to **CoreML** and run it on the Apple Neural Engine. **Performance/latency is the
overriding priority** — see [`PLAN.md`](./PLAN.md).

## Toolchain (Python, conversion side)

Fully self-contained via [`uv`](https://docs.astral.sh/uv/) — pinned to Python 3.11, project-local
`.venv`, locked deps in `uv.lock`.

```bash
make setup     # uv sync — create/sync the venv (all dependency groups)
make models    # download official BlazePose .tflite models -> models/tflite/
make inspect   # dump each model's I/O contract -> models/tflite/io_contract.json
make verify    # parity test: PyTorch port vs every .tflite reference
make lint      # ruff
```

The tflite→PyTorch port lives in `poseml.tflite_port` — `build("models/tflite/pose_landmark_full.tflite")`
returns a traceable `nn.Module` numerically equivalent to the .tflite (no separate checkpoint).

Dependency groups (in `pyproject.toml`):
- **core** — `coremltools`, `torch`, `numpy`, `opencv`, `pillow` (port + convert)
- **reference** — `ai-edge-litert` (run original `.tflite` for parity checks only)
- **inspect** — `netron`
- **dev** — `ruff`, `pytest`

## Layout

```
models/{tflite,pytorch,coreml}/   model binaries (not committed; reproduced via make)
python/poseml/{convert,verify,anchors}/   conversion toolchain (importable as `poseml`)
ios/                              Swift app + on-device pipeline
PLAN.md                           full implementation plan
```

## Status
- **Phase 0 (env + assets + I/O contract): done.** See `models/tflite/io_contract.json`.
- **Phase 1 (PyTorch parity): done.** Generic tflite→PyTorch port; all 4 models pass `make verify`.
- Next: Phase 2 — CoreML conversion (trace → coremltools, FP16/ANE, prune mask+heatmap).
