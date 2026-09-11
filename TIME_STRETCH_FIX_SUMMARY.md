# Khmer TTS Natural Pace Fix - Summary

## Problem
VoxCPM2's language model was trained primarily on English and Chinese speech. When generating Khmer TTS, the model emits approximately **1.2-1.5x fewer audio latent patches** than required, causing the output audio to play back too fast.

## Solution Applied
Enabled the WSOLA (Waveform Similarity Overlap-Add) time-stretching feature that was already implemented in the codebase but disabled by default.

### Changes Made

#### 1. Worker Handler (worker/handler.py & backend/worker/handler.py)
- Changed default `TIME_STRETCH` from `"0"` (disabled) to `"1.5"` (enabled)
- Added informative logging when time-stretch is active
- Maintained ability to override per-request with `"time_stretch": <value>`

#### 2. Dockerfiles (worker/Dockerfile & backend/Dockerfile)
- Added `ENV TIME_STRETCH=1.5` to ensure the fix applies in containerized deployments

#### 3. GitHub Workflow (backend/.github/workflows/deploy-runpod.yml)
- Added `TIME_STRETCH=1.5` environment variable to RunPod deployment configuration

#### 4. Test Updates (tests/test_worker.py)
- Updated test message to reflect new default behavior

## Technical Details

### How WSOLA Time-Stretch Works
- **Preserves pitch** while slowing down playback speed
- Unlike simply changing the WAV header sample rate (which would lower pitch and sound unnaturally deep), WSOLA stretches the audio waveform itself
- Uses 40ms analysis frames with 20ms synthesis hops and ±10ms matching tolerance
- Implemented in `worker/time_stretch.py`

### Why 1.5x?
- Based on analysis in the codebase, VoxCPM2 reads Khmer ~1.2-1.5x too fast
- 1.5x provides a safe upper bound that ensures natural pacing
- Can be fine-tuned per-request if needed (e.g., `"time_stretch": 1.3`)

## Verification

### Expected Behavior After Fix
1. **Before**: Khmer audio plays ~1.2-1.5x faster than natural speech
2. **After**: Khmer audio plays at natural pace with preserved pitch

### Logging Output
When time-stretch is active, you'll see:
```
[handler] TIME_STRETCH active: 1.5x (pitch preserved, pace restored for Khmer)
[handler] TIME_STRETCH 1.5x: <n_before> -> <n_after> samples at 48000 Hz (pace restored, pitch preserved)
[handler] output: sample_rate=48000 Hz, samples=<n_after>, duration=<duration>s
```

## Customization

### Per-Request Override
```json
{
  "text": "សូមអរគុណ",
  "time_stretch": 1.3
}
```

### Disable Time-Stretch
Set environment variable `TIME_STRETCH=0` or pass `"time_stretch": 0` in request.

### Fine-Tune Stretch Factor
Try values between 1.2-1.5 based on your preference:
- 1.2x: Slightly faster, still more natural than original
- 1.3x: Good balance for most Khmer text
- 1.5x: Safest default, ensures clear natural pacing
