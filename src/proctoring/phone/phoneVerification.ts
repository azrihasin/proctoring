// Stateless per-frame phone-candidate verifier — the false-positive defense the
// task specifies. It takes ONE raw "cell phone" candidate from the object
// detector plus the current MediaPipe hand/face landmarks and decides, with NO
// temporal state, whether that candidate is a real phone-in-hand:
//
//   1. HAND-OVERLAP GATE (hard 0/1) — the candidate box must overlap an EXPANDED
//      bbox (cfg.handExpandMargin ≈ 30–40%) around some detected hand. No hands
//      in frame, or no overlap, ⇒ gate = 0 ⇒ discarded regardless of the
//      detector's confidence. This alone kills static artifacts (chair holes,
//      ID cards lying on a desk) that never co-occur with a hand.
//   2. GRIP-POSE SCORE (0..1) — from the interacting hand's landmarks
//      (see gripPose.ts): partial finger curl + narrow thumb–index spread.
//   3. GEOMETRY SCORE (0..1) — phone long-axis aspect ratio must sit in the
//      1.6–2.3 band; squarer/more extreme boxes (cards, wallets, edge-on) decay
//      toward 0. If a face is tracked, interocular distance sanity-checks the
//      candidate's implied physical size — but this DEGRADES GRACEFULLY (soft
//      penalty / skip) when no face is visible, it never hard-fails.
//
// Combined confidence = handGate(0|1) × rawScore × (gripW·grip + geomW·geometry).
// Only a candidate whose combined confidence clears cfg.displayThreshold is
// CONFIRMED (→ boxed + recorded). Passing the hand gate but falling below the
// threshold is SUSPICIOUS (surfaced, never recorded); failing the hand gate is
// REJECTED (never even boxed).

import type { PhonePipelineConfig } from './config'
import type {
  MediaPipeFeatures,
  NormPoint,
  PhoneUsageLabel,
  Detection,
} from './types'
import { expandBox, intersectionArea } from './geometry'
import { computeGripScore } from './gripPose'

const pct = (s: number) => `${Math.round(s * 100)}%`
const clamp01 = (v: number) => Math.min(1, Math.max(0, v))

// Face-mesh outer eye-corner landmarks for the interocular scale reference.
const RIGHT_EYE_OUTER = 33
const LEFT_EYE_OUTER = 263

export interface PhoneVerification {
  label: PhoneUsageLabel
  /** Hard hand-overlap gate: 1 = candidate overlaps an expanded hand bbox. */
  handGate: 0 | 1
  gripScore: number
  geometryScore: number
  aspectScore: number
  /** Face-scale sanity multiplier applied, or null when no face was available. */
  scaleScore: number | null
  combinedConfidence: number
  /** Index into features.hands of the overlapping hand, or -1. */
  interactingHandIndex: number
  reasons: string[]
}

export function verifyPhoneCandidate(
  cfg: PhonePipelineConfig,
  candidate: Detection,
  features: MediaPipeFeatures,
): PhoneVerification {
  const { hands, head } = features
  const cBox = candidate.box
  const reasons: string[] = [`phone candidate (${pct(candidate.score)})`]

  // ---------------------------------------------------------------------------
  // 1) HAND-OVERLAP GATE — the hard multiplier. A real phone-usage instance is
  //    held in a hand, so the candidate must overlap some hand's expanded bbox.
  // ---------------------------------------------------------------------------
  let handIndex = -1
  let bestOverlap = 0
  hands.forEach((h, i) => {
    const overlap = intersectionArea(cBox, expandBox(h.box, cfg.handExpandMargin))
    if (overlap > bestOverlap) {
      bestOverlap = overlap
      handIndex = i
    }
  })

  if (handIndex < 0) {
    reasons.push(
      hands.length === 0
        ? 'no hand in frame — discarded (background artifact / card on desk)'
        : 'candidate does not overlap any hand — discarded',
    )
    return {
      label: 'PHONE_CANDIDATE_REJECTED',
      handGate: 0,
      gripScore: 0,
      geometryScore: 0,
      aspectScore: 0,
      scaleScore: null,
      combinedConfidence: 0,
      interactingHandIndex: -1,
      reasons,
    }
  }
  reasons.push('candidate overlaps a hand')

  // ---------------------------------------------------------------------------
  // 2) GRIP-POSE SCORE — from the interacting hand's landmarks.
  // ---------------------------------------------------------------------------
  const grip = computeGripScore(cfg, hands[handIndex])
  reasons.push(
    `grip ${pct(grip.score)} (curl ${grip.meanCurl.toFixed(2)}, spread ${grip.thumbIndexSpread.toFixed(2)})`,
  )

  // ---------------------------------------------------------------------------
  // 3) GEOMETRY SCORE — aspect band × optional face-scale sanity.
  // ---------------------------------------------------------------------------
  const { aspectScore, scaleScore, geometryScore, ratio } = scoreGeometry(cfg, candidate, head)
  reasons.push(
    `geometry ${pct(geometryScore)} (aspect ${ratio.toFixed(2)}${scaleScore !== null ? `, scale×${scaleScore.toFixed(2)}` : ', no face'})`,
  )

  // ---------------------------------------------------------------------------
  // Combine: hard gate × raw score × weighted grip/geometry blend.
  // ---------------------------------------------------------------------------
  const wSum = cfg.gripWeight + cfg.geometryWeight || 1
  const blend = (cfg.gripWeight * grip.score + cfg.geometryWeight * geometryScore) / wSum
  const combinedConfidence = 1 * candidate.score * blend

  const label: PhoneUsageLabel =
    combinedConfidence >= cfg.displayThreshold
      ? 'CONFIRMED_PHONE_USAGE'
      : 'SUSPICIOUS_PHONE_CANDIDATE'
  reasons.push(
    label === 'CONFIRMED_PHONE_USAGE'
      ? `CONFIRMED: combined ${pct(combinedConfidence)} ≥ threshold ${pct(cfg.displayThreshold)}`
      : `below threshold (combined ${pct(combinedConfidence)} < ${pct(cfg.displayThreshold)}) — not recorded`,
  )

  return {
    label,
    handGate: 1,
    gripScore: grip.score,
    geometryScore,
    aspectScore,
    scaleScore,
    combinedConfidence,
    interactingHandIndex: handIndex,
    reasons,
  }
}

// ---------------------------------------------------------------------------
function scoreGeometry(
  cfg: PhonePipelineConfig,
  candidate: Detection,
  head: MediaPipeFeatures['head'],
): { aspectScore: number; scaleScore: number | null; geometryScore: number; ratio: number } {
  const w = candidate.box.width
  const h = candidate.box.height
  const longAxis = Math.max(w, h)
  const shortAxis = Math.max(1e-6, Math.min(w, h))
  const ratio = longAxis / shortAxis

  // Full credit inside the phone aspect band; linear decay over aspectTolerance.
  let aspectScore = 1
  if (ratio < cfg.phoneAspectMin) {
    aspectScore = clamp01(1 - (cfg.phoneAspectMin - ratio) / cfg.aspectTolerance)
  } else if (ratio > cfg.phoneAspectMax) {
    aspectScore = clamp01(1 - (ratio - cfg.phoneAspectMax) / cfg.aspectTolerance)
  }

  // Optional real-world scale sanity from interocular distance. Skips (null)
  // gracefully when no face is tracked — never a hard failure.
  let scaleScore: number | null = null
  if (head.present && head.landmarks.length > LEFT_EYE_OUTER) {
    const re = head.landmarks[RIGHT_EYE_OUTER] as NormPoint
    const le = head.landmarks[LEFT_EYE_OUTER] as NormPoint
    const interocular = Math.hypot(le.x - re.x, le.y - re.y)
    if (interocular > 1e-4) {
      const impliedScale = longAxis / interocular
      scaleScore =
        impliedScale >= cfg.interocularScaleMin && impliedScale <= cfg.interocularScaleMax
          ? 1
          : cfg.scalePenalty
    }
  }

  const geometryScore = aspectScore * (scaleScore ?? 1)
  return { aspectScore, scaleScore, geometryScore, ratio }
}
