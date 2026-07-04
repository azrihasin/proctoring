// Confirmation / rejection rule engine — the heart of the false-positive
// defense. It NEVER treats a raw "cell phone" detection as a violation. Instead
// it runs every phone_candidate through four evidence gates and a set of
// rejection rules, and emits exactly one of:
//
//   NO_VIOLATION
//   PHONE_CANDIDATE_REJECTED
//   SUSPICIOUS_PHONE_CANDIDATE
//   CONFIRMED_PHONE_USAGE   ← the only label that should trigger recording
//
// Gates (all four required for CONFIRMED_PHONE_USAGE):
//   1. person association — candidate is on/near the student, not the wall/chair
//   2. hand interaction   — candidate is at a hand (wrist/palm/fingers/between)
//   3. head / gaze        — student is looking down / toward the candidate
//   4. suspicious zone    — candidate or interacting hand is low in the frame
//
// Rejections shed the classic false positives (cards, chair-holes, background,
// forward-facing high holds). Thresholds are all read from config so behavior
// can be tuned without editing this logic.

import type { PhonePipelineConfig } from './config'
import type {
  GateResults,
  HandFeature,
  MediaPipeFeatures,
  NormBox,
  NormPoint,
  PhoneUsageDecision,
  PoseFeature,
  DetectionFrame,
} from './types'
import type { WebcamZones } from './webcamZones'
import { boxCenter, boxIoU, dist, pointInBox } from './geometry'
import { verifyPhoneCandidate } from './phoneVerification'

const pct = (s: number) => `${Math.round(s * 100)}%`

export function classifyPhoneUsage(
  cfg: PhonePipelineConfig,
  frame: DetectionFrame,
  features: MediaPipeFeatures,
  zones: WebcamZones,
): PhoneUsageDecision {
  const { head, pose, hands } = features
  const candidate = frame.phoneCandidates[0] ?? null

  // -------------------------------------------------------------------------
  // No phone candidate → at most "suspicious behavior", never confirmed usage.
  // -------------------------------------------------------------------------
  if (!candidate) {
    const handsHiddenOrLow = areHandsHiddenOrLow(cfg, hands, pose, zones)
    const gates = emptyGates()
    if (head.headDown && handsHiddenOrLow) {
      return {
        label: 'SUSPICIOUS_PHONE_CANDIDATE',
        candidate: null,
        gates,
        passedGates: [],
        reasons: [
          'no phone candidate detected',
          'head looking down',
          'hands hidden or held low',
          'suspicious behavior only — not confirmed phone usage',
        ],
        interactingHandIndex: -1,
        combinedConfidence: 0,
        gripScore: 0,
        geometryScore: 0,
      }
    }
    return {
      label: 'NO_VIOLATION',
      candidate: null,
      gates,
      passedGates: [],
      reasons: ['no phone candidate detected'],
      interactingHandIndex: -1,
      combinedConfidence: 0,
      gripScore: 0,
      geometryScore: 0,
    }
  }

  const cBox = candidate.box
  const cCenter = boxCenter(cBox)
  const reasons: string[] = [`phone candidate detected (${pct(candidate.score)})`]

  // ---- Gate 1: person / spatial association ----
  const person = frame.person
  let personAssociation = false
  if (person) {
    const overlap = boxIoU(cBox, person.box)
    const centerNear = pointInBox(cCenter, person.box, cfg.personProximityMargin)
    personAssociation = overlap > 0.001 || centerNear
    if (personAssociation) reasons.push('candidate on/near the student')
    else reasons.push('candidate outside the student area (background/chair/wall)')
  } else {
    reasons.push('no person box to associate candidate with')
  }

  // ---- Gate 2: hand interaction ----
  const hi = evaluateHandInteraction(cfg, cBox, cCenter, hands)
  const handInteraction = hi.interacts
  if (handInteraction) reasons.push(hi.reason)
  else reasons.push('candidate far from both hands')

  // ---- Gate 3: head / gaze ----
  const faceTowardCandidate =
    head.present && head.noseTip != null &&
    Math.abs(head.noseTip.x - cCenter.x) < cfg.faceTowardCandidateDx
  const headGaze =
    head.headDown ||
    (head.turnedAway && faceTowardCandidate) ||
    (faceTowardCandidate && head.gazeDown >= cfg.gazeDownThreshold)
  if (head.headDown) reasons.push('head looking down')
  else if (head.turnedAway && faceTowardCandidate) reasons.push('face turned toward candidate')
  else if (faceTowardCandidate) reasons.push('gaze/face oriented toward candidate')
  else reasons.push('face forward — not oriented toward candidate')

  // ---- Gate 4: suspicious webcam zone ----
  const candidateLow = zones.isBoxSuspicious(cBox)
  const interactingHandLow =
    hi.handIndex >= 0 && zones.isPointSuspicious(hands[hi.handIndex].wrist)
  const poseWristLow = isAnyPoseWristLow(cfg, pose, zones)
  const suspiciousZone = candidateLow || interactingHandLow || poseWristLow
  if (candidateLow) reasons.push('candidate in lower/bottom webcam zone')
  else if (interactingHandLow) reasons.push('interacting hand is low in the frame')
  else if (poseWristLow) reasons.push('a wrist is low/near the bottom of the frame')
  else reasons.push('candidate held high in the frame')

  const gates: GateResults = { personAssociation, handInteraction, headGaze, suspiciousZone }

  // -------------------------------------------------------------------------
  // Decision — the STATELESS VERIFIER owns the label. Per the false-positive
  // fix it treats every "cell phone" box as a candidate and gates it through a
  // hard hand-overlap check, a grip-pose score and a box-geometry score,
  // combining them into a single confidence:
  //   handGate(0|1) × rawScore × (gripWeight·grip + geometryWeight·geometry).
  // CONFIRMED only when that clears cfg.displayThreshold. The four gates above
  // (person / hand / head / zone) are retained purely as explainable
  // diagnostics for the overlay and captured evidence — they do NOT decide the
  // label, and there is NO temporal smoothing here.
  // -------------------------------------------------------------------------
  const verdict = verifyPhoneCandidate(cfg, candidate, features)

  return {
    label: verdict.label,
    candidate,
    gates,
    passedGates: passedGateNames(gates),
    reasons: [...reasons, ...verdict.reasons],
    interactingHandIndex: verdict.interactingHandIndex,
    combinedConfidence: verdict.combinedConfidence,
    gripScore: verdict.gripScore,
    geometryScore: verdict.geometryScore,
  }
}

// ---------------------------------------------------------------------------
// Hand interaction
// ---------------------------------------------------------------------------
export interface HandInteraction {
  interacts: boolean
  handIndex: number
  reason: string
}

export function evaluateHandInteraction(
  cfg: PhonePipelineConfig,
  cBox: NormBox,
  cCenter: NormPoint,
  hands: HandFeature[],
): HandInteraction {
  let bestIdx = -1
  let bestDist = Infinity

  hands.forEach((h, i) => {
    // Nearest of wrist / palm / fingertips to the candidate center.
    const pts = [h.wrist, h.palmCenter, ...h.fingertips]
    const d = Math.min(...pts.map((p) => dist(p, cCenter)))
    const overlap = boxIoU(cBox, h.box)
    const near = d < cfg.handProximityDist || overlap >= cfg.handOverlapIoU
    if (near && d < bestDist) { bestDist = d; bestIdx = i }
  })

  if (bestIdx >= 0) {
    const side = handSide(hands[bestIdx], bestIdx)
    return { interacts: true, handIndex: bestIdx, reason: `candidate at ${side} hand` }
  }

  // "Between both hands"
  if (hands.length >= 2) {
    const [a, b] = hands
    const minX = Math.min(a.wrist.x, b.wrist.x)
    const maxX = Math.max(a.wrist.x, b.wrist.x)
    const midY = (a.wrist.y + b.wrist.y) / 2
    if (cCenter.x >= minX && cCenter.x <= maxX && Math.abs(cCenter.y - midY) <= cfg.betweenHandsBand) {
      return { interacts: true, handIndex: 0, reason: 'candidate held between both hands' }
    }
  }

  return { interacts: false, handIndex: -1, reason: 'candidate far from both hands' }
}

function handSide(h: HandFeature, idx: number): string {
  if (h.handedness === 'Left') return 'left'
  if (h.handedness === 'Right') return 'right'
  return `hand ${idx + 1}`
}

// ---------------------------------------------------------------------------
// Low-wrist / hidden-hands helpers
// ---------------------------------------------------------------------------
function isAnyPoseWristLow(
  cfg: PhonePipelineConfig,
  pose: PoseFeature | null,
  zones: WebcamZones,
): boolean {
  if (!pose) return false
  const chest = pose.chestLineY ?? zones.chestLineY
  const lowThreshold = chest + cfg.lowWristMargin
  const check = (w: NormPoint | null) => w != null && (w.y >= lowThreshold || zones.isPointSuspicious(w))
  return check(pose.leftWrist) || check(pose.rightWrist)
}

function areHandsHiddenOrLow(
  cfg: PhonePipelineConfig,
  hands: HandFeature[],
  pose: PoseFeature | null,
  zones: WebcamZones,
): boolean {
  // No hands detected at all → hidden.
  if (hands.length === 0) {
    // If pose is present, a missing/low wrist strengthens "hidden/low".
    if (pose) {
      const wristMissing = pose.leftWrist == null || pose.rightWrist == null
      return wristMissing || isAnyPoseWristLow(cfg, pose, zones)
    }
    return true
  }
  // Hands present but all in the suspicious (low) zone.
  return hands.every((h) => zones.isPointSuspicious(h.wrist))
}

// ---------------------------------------------------------------------------
function emptyGates(): GateResults {
  return { personAssociation: false, handInteraction: false, headGaze: false, suspiciousZone: false }
}

function passedGateNames(g: GateResults): string[] {
  const out: string[] = []
  if (g.personAssociation) out.push('personAssociation')
  if (g.handInteraction) out.push('handInteraction')
  if (g.headGaze) out.push('headGaze')
  if (g.suspiciousZone) out.push('suspiciousZone')
  return out
}
