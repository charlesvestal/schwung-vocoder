# move-anything-vocoder

## AI Assistance Disclaimer

This module is part of Move Everything and was developed with AI assistance, including Claude, Codex, and other AI assistants.

All architecture, implementation, and release decisions are reviewed by human maintainers.  
AI-assisted content may still contain errors, so please validate functionality, security, and license compatibility before production use.

## How it works

A classic channel vocoder: the **modulator** (mic / line-in, read from the
hardware input buffer) is split into bands by a stereo state-variable
filterbank; its per-band envelopes shape the **carrier** (the synth output
arriving from the signal chain). Put a bright, sustained synth before this
effect and speak/sing into the mic.

## Parameters

Core: **Bands** (8/16/24/32), **Low/High Freq** (filterbank range),
**Attack/Release** (envelope tracking), **Mod Gain**, **Out Gain**, **Mix**
(wet/dry), **Unvoiced** (`carrier_mix` — noise added to the carrier).

Intelligibility / character controls. **Presence and Bright ship on** (tuned
for clear vocals out of the box) — set them to 0 for the original "classic"
character. The situational controls (Sibilance, Gate, Note Gate, Formant)
default to off/neutral.

| Param | Range | What it does |
|-------|-------|--------------|
| **Presence** | 0–1 | Pre-emphasises the voice's highs before analysis so consonants track better. |
| **Sibilance** | 0–1 | Passes the voice's own high-frequency content through (natural "s/t/f"), instead of relying only on carrier noise. This is a direct mic→output path: with an open speaker + mic it can feed back, so raise it carefully and consider **Note Gate**. |
| **Bright** | 0–1 | Lifts the carrier's highs so upper formants survive. |
| **Formant** | −12…+12 st | Shifts the spectral envelope up/down by that many semitones (±1 octave), interpolated across bands — the interval per step is the same at any band count. |
| **Gate** | 0–1 | Noise gate on the voice — silences room hiss/hum between words (0 = off). |
| **Note Gate** | 0–1 | Mutes the unvoiced noise + sibilance when no carrier is present, so the effect is silent between notes (0 = off, preserves the always-on behaviour). |

**Defaults** are Presence 0.3, Bright 0.3, Bands 16 — a good clear-vocal
starting point. From there, raise **Sibilance** for natural "s/t/f" (watch
levels on open speaker+mic setups), raise **Gate** until the background
between words goes quiet, and try **Formant** for character.
