// Grip-pose scorer — turns one hand's 21 MediaPipe landmarks into a single
// 0..1 "does this look like a hand gripping a phone" score, statelessly.
//
// The intuition the task describes:
//   • A genuine phone grip → two or three fingers PARTIALLY curl around an edge
//     while the thumb sits close to the index (a NARROW thumb–index spread).
//   • A flat card / open palm presented to the camera → fingers extended nearly
//     flat (little curl) with a WIDE, open thumb–index spread.
//
// So the score rewards moderate finger curl AND a narrow thumb–index gap. Both
// sub-scores and every threshold are config-driven so the grip model can be
// retuned without touching the verifier that consumes it.

import type { HandFeature, NormPoint } from './types'
import type { PhonePipelineConfig } from './config'

// Landmark indices (MediaPipe Hand): each finger is MCP → PIP → DIP → TIP.
const FINGERS = [
  { name: 'index', mcp: 5, pip: 6, tip: 8 },
  { name: 'middle', mcp: 9, pip: 10, tip: 12 },
  { name: 'ring', mcp: 13, pip: 14, tip: 16 },
  { name: 'pinky', mcp: 17, pip: 18, tip: 20 },
] as const
const THUMB_TIP = 4
const INDEX_TIP = 8
const MIDDLE_MCP = 9
const WRIST = 0

// Extended finger ≈ 180° at the PIP joint; a fully folded finger ≈ 40°. Curl is
// mapped linearly across that span, so a partially wrapped finger reads ~0.4–0.7.
const EXTENDED_ANGLE = 180
const FOLDED_ANGLE = 40

export interface GripScore {
  /** Combined grip likelihood, 0..1. */
  score: number
  /** Mean finger curl across index/middle/ring/pinky, 0..1. */
  meanCurl: number
  /** Thumb-tip↔index-tip distance ÷ palm length (hand-scale invariant). */
  thumbIndexSpread: number
  /** Per-finger curl for evidence/debug. */
  perFingerCurl: number[]
}

/** 3D vector from a→b (z defaults to 0 when a landmark lacks depth). */
function vec(a: NormPoint, b: NormPoint) {
  return { x: b.x - a.x, y: b.y - a.y, z: (b.z ?? 0) - (a.z ?? 0) }
}

/** Angle (degrees) at the PIP joint between the MCP and TIP directions. */
function jointAngleDeg(mcp: NormPoint, pip: NormPoint, tip: NormPoint): number {
  const a = vec(pip, mcp)
  const b = vec(pip, tip)
  const dot = a.x * b.x + a.y * b.y + a.z * b.z
  const na = Math.hypot(a.x, a.y, a.z)
  const nb = Math.hypot(b.x, b.y, b.z)
  if (na === 0 || nb === 0) return EXTENDED_ANGLE
  const cos = Math.min(1, Math.max(-1, dot / (na * nb)))
  return (Math.acos(cos) * 180) / Math.PI
}

const clamp01 = (v: number) => Math.min(1, Math.max(0, v))

export function computeGripScore(cfg: PhonePipelineConfig, hand: HandFeature): GripScore {
  const lm = hand.landmarks

  // --- Finger curl (0 = extended, 1 = folded) for the four non-thumb fingers ---
  const perFingerCurl = FINGERS.map((f) => {
    const angle = jointAngleDeg(lm[f.mcp], lm[f.pip], lm[f.tip])
    return clamp01((EXTENDED_ANGLE - angle) / (EXTENDED_ANGLE - FOLDED_ANGLE))
  })
  const meanCurl = perFingerCurl.reduce((s, c) => s + c, 0) / perFingerCurl.length

  // A grip needs *partial* curl: flat hands score 0, moderate wrap scores 1.
  const curlScore = clamp01(
    (meanCurl - cfg.gripFlatCurl) / Math.max(1e-6, cfg.gripFullCurl - cfg.gripFlatCurl),
  )

  // --- Thumb–index spread, normalized by palm length so it's scale-invariant ---
  const palmLen = Math.max(1e-6, Math.hypot(lm[MIDDLE_MCP].x - lm[WRIST].x, lm[MIDDLE_MCP].y - lm[WRIST].y))
  const thumbIndexSpread =
    Math.hypot(lm[THUMB_TIP].x - lm[INDEX_TIP].x, lm[THUMB_TIP].y - lm[INDEX_TIP].y) / palmLen

  // Narrow spread (wrapping an edge) → 1; wide-open palm/card → 0.
  const spreadScore = clamp01(
    (cfg.gripMaxSpread - thumbIndexSpread) / Math.max(1e-6, cfg.gripMaxSpread - cfg.gripMinSpread),
  )

  const wSum = cfg.gripCurlWeight + cfg.gripSpreadWeight || 1
  const score = (cfg.gripCurlWeight * curlScore + cfg.gripSpreadWeight * spreadScore) / wSum

  return { score, meanCurl, thumbIndexSpread, perFingerCurl }
}
