"""Pitch-preserving time-stretch (WSOLA) for VoxCPM2 output.

Root cause this fixes: VoxCPM2's language model was trained mainly on English
and Chinese speech. For out-of-domain scripts such as Khmer it emits roughly
1.2-1.5x fewer audio latent patches than the text duration requires, so the
decoded 48 kHz waveform is speech read proportionally too fast. The audio
chain around it (WAV header, sample rate, player, API proxy) is correct.

Stretching the decoded waveform with WSOLA restores the natural reading pace
while keeping the voice's pitch unchanged — a WAV-header slow-down would also
lower the pitch, which sounds unnaturally deep.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def wsola_time_stretch(wav: np.ndarray, sample_rate: int, factor: float) -> np.ndarray:
    """Time-stretch `wav` by `factor` (1.5 -> output 1.5x longer) with WSOLA.

    Pitch is preserved: only the reading pace changes. Mono float32 in/ out.
    """
    if factor <= 1.0 + 1e-6 or wav.size == 0:
        return np.asarray(wav, dtype=np.float32).reshape(-1)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    sr = int(sample_rate)
    N = int(0.040 * sr)    # 40 ms analysis frame
    if wav.size <= N + 1:
        # Input shorter than one analysis frame: plain repetition is the only
        # sensible stretch (WSOLA overlap-add needs at least one full frame).
        return np.tile(wav, max(1, int(round(factor))))
    Hs = int(0.020 * sr)   # 20 ms synthesis hop (50% overlap with Hann window)
    Ha = max(1, int(round(Hs / factor)))  # analysis hop = synthesis hop / factor
    L = int(0.010 * sr)    # +/-10 ms WSOLA match tolerance
    win = np.hanning(N).astype(np.float32)
    n_out = int(wav.size * factor) + N
    y = np.zeros(n_out, dtype=np.float32)
    norm = np.zeros(n_out, dtype=np.float32)
    out_pos = 0
    pos = 0
    prev: np.ndarray | None = None
    while True:
        if prev is not None:
            # WSOLA: shift the analysis position so the frame's natural
            # continuation aligns best with what the previous output frame
            # promised (within +/-L samples). Keeps output click-free.
            lo = max(0, pos + Ha - L)
            hi = min(wav.size - L - 1, pos + Ha + L)
            if hi > lo:
                windows = sliding_window_view(wav[lo:hi + L + 1], L)
                scores = windows @ prev
                pos = lo + int(np.argmax(scores))
            else:
                pos = pos + Ha
        if pos + N >= wav.size or out_pos + N >= n_out:
            break
        frame = wav[pos:pos + N]
        y[out_pos:out_pos + N] += frame * win
        norm[out_pos:out_pos + N] += win
        prev = wav[pos + N:pos + N + L].copy()
        if prev.size < L:
            break
        out_pos += Hs
    # Flush the trailing audio that the frame loop skipped (at most one frame).
    tail = wav[pos:]
    if tail.size and out_pos < n_out:
        end = min(n_out, out_pos + tail.size)
        y[out_pos:end] += tail[:end - out_pos]
        norm[out_pos:end] += 1.0
    y /= np.maximum(norm, 1e-9)
    return y[:out_pos]
