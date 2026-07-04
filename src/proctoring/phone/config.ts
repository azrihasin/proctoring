// Central tunables for the object-detector + MediaPipe phone-usage pipeline.
//
// Everything the rule engine, zone calculator, detector, and hook reason about
// is a named constant here so thresholds (hand distance, gaze/head angle,
// lower-frame zone, overlay visibility, rejection rules) can be tuned later
// WITHOUT touching pipeline logic — which is the whole point of the modular
// split the task asked for.

export interface PhonePipelineConfig {
  // ---- Detection cadence ----
  intervalMs: number

  // ---- Candidate detection ----
  /** Min confidence for a "cell phone" box to be surfaced as a CANDIDATE. */
  candidateScoreThreshold: number
  /** Min confidence for a person box to be trusted for spatial association. */
  personScoreThreshold: number
  /** Min confidence for generic object boxes drawn on the overlay. */
  objectScoreThreshold: number

  // ---- Gate 1: person / spatial association ----
  /**
   * A candidate is "associated with the student" if it overlaps the person box
   * OR its center is within this margin (fraction of frame) of the person box.
   * Prevents phones on the wall/chair/background counting.
   */
  personProximityMargin: number

  // ---- Gate 2: hand interaction ----
  /**
   * Candidate is "near a hand" if the distance from the candidate center to any
   * hand landmark (wrist/palm/fingertip) is below this (fraction of frame diag).
   */
  handProximityDist: number
  /** Candidate also counts as hand-interacting if IoU with a hand box ≥ this. */
  handOverlapIoU: number
  /**
   * "Between both hands": candidate center x lies between the two wrists and its
   * y is within this vertical band (fraction of frame) of the wrist midline.
   */
  betweenHandsBand: number

  // ---- Gate 3: head / gaze ----
  /** Head pitch (deg, +down) at/above this = looking down. */
  headPitchDownDeg: number
  /** Gaze-down blendshape at/above this = eyes below the screen. */
  gazeDownThreshold: number
  /** |yaw| at/above this = face turned away from forward. */
  headYawAwayDeg: number
  /**
   * "Face oriented toward candidate": horizontal offset between nose tip and
   * candidate center below this (fraction of frame) while head is down.
   */
  faceTowardCandidateDx: number

  // ---- Gate 4: suspicious webcam zone ----
  /** Bottom fraction of the FRAME treated as a suspicious zone (near edge). */
  bottomFrameZone: number
  /** Bottom fraction of the PERSON box treated as suspicious (lower torso). */
  bottomPersonZone: number
  /**
   * A wrist is "low" (suspicious) if its y is below chestLineY + this fraction,
   * or in the bottom-frame zone, or missing/occluded.
   */
  lowWristMargin: number

  // ---- Rejection rules ----
  /** Reject a candidate larger than this frame fraction (card/laptop/monitor). */
  cardMaxAreaFraction: number
  /** Reject flat, card-like candidates wider than tall beyond this aspect. */
  cardMaxAspect: number
  /**
   * If candidate is high (above chest line by this margin) AND head faces
   * forward AND both hands visible, reject as "held high, facing forward".
   */
  heldHighMargin: number

  // ---- Stateless hand-overlap + grip + geometry verifier (per-frame) ----
  /**
   * Fraction the raw hand bbox is expanded on every side before testing overlap
   * with a phone candidate (the task's ~30–40% hand margin). The hand-overlap
   * gate is a HARD 0/1 multiplier: no overlap ⇒ candidate discarded outright.
   */
  handExpandMargin: number
  /** Phone long-axis aspect-ratio band, lower bound (long ÷ short). */
  phoneAspectMin: number
  /** Phone long-axis aspect-ratio band, upper bound. */
  phoneAspectMax: number
  /** How far outside the aspect band the geometry score decays to 0. */
  aspectTolerance: number
  /** meanCurl (0..1) at/below which fingers read as flat/extended (card/palm). */
  gripFlatCurl: number
  /** meanCurl at/above which fingers read as a full grip. */
  gripFullCurl: number
  /** thumb-index tip spread (÷ palm length) at/below which the grip is tight. */
  gripMinSpread: number
  /** thumb-index spread at/above which the hand is wide open (card/palm). */
  gripMaxSpread: number
  /** Grip sub-score weights: finger curl vs. thumb-index spread. */
  gripCurlWeight: number
  gripSpreadWeight: number
  /** Combined-confidence blend weights: grip vs. geometry (sum ≈ 1). */
  gripWeight: number
  geometryWeight: number
  /**
   * Tunable DISPLAY threshold. Combined confidence
   * (handGate × rawScore × grip/geometry blend) must clear this for a candidate
   * to be boxed on screen AND recorded as a violation.
   */
  displayThreshold: number
  /** Plausible candidate-long-axis ÷ interocular-distance band (face scale check). */
  interocularScaleMin: number
  interocularScaleMax: number
  /** Soft multiplier applied to geometry when the implied size is implausible. */
  scalePenalty: number

  // ---- Confirmation persistence / hysteresis ----
  /**
   * Instant detection: fire CONFIRMED_PHONE_USAGE on the FIRST qualifying tick,
   * bypassing the consecutive-tick persistence gate below. The task requires the
   * phone-usage gesture to register instantly with no temporal smoothing. Set
   * false to fall back to the persistence-based confirmation.
   */
  instantConfirm: boolean
  /** Consecutive ticks a CONFIRMED_PHONE_USAGE label must persist to fire. */
  requiredConsecutive: number
  /** Boosted (lower) requirement when head-down strongly corroborates. */
  requiredConsecutiveBoosted: number
  /** IoU a confirming candidate must keep with the previous tick's candidate. */
  persistIoU: number
  /** Consecutive missing ticks before an open incident clears (hysteresis). */
  clearMisses: number

  // ---- Overlay visibility toggles ----
  overlay: {
    detectionBoxes: boolean
    faceMesh: boolean
    faceContours: boolean
    pose: boolean
    hands: boolean
    zones: boolean
    labelBanner: boolean
  }

  // ---- Feature toggles ----
  enableHands: boolean
  enablePose: boolean

  debug: boolean
}

export const DEFAULT_PHONE_CONFIG: PhonePipelineConfig = {
  intervalMs: 140,

  candidateScoreThreshold: 0.4,
  personScoreThreshold: 0.5,
  objectScoreThreshold: 0.45,

  personProximityMargin: 0.06,

  handProximityDist: 0.14,
  handOverlapIoU: 0.02,
  betweenHandsBand: 0.18,

  headPitchDownDeg: 12,
  gazeDownThreshold: 0.35,
  headYawAwayDeg: 22,
  faceTowardCandidateDx: 0.18,

  bottomFrameZone: 0.3,
  bottomPersonZone: 0.3,
  lowWristMargin: 0.22,

  cardMaxAreaFraction: 0.14,
  cardMaxAspect: 1.9,
  heldHighMargin: 0.05,

  handExpandMargin: 0.35,
  phoneAspectMin: 1.6,
  phoneAspectMax: 2.3,
  aspectTolerance: 0.8,
  gripFlatCurl: 0.15,
  gripFullCurl: 0.45,
  gripMinSpread: 0.25,
  gripMaxSpread: 0.9,
  gripCurlWeight: 0.6,
  gripSpreadWeight: 0.4,
  gripWeight: 0.5,
  geometryWeight: 0.5,
  displayThreshold: 0.35,
  interocularScaleMin: 0.8,
  interocularScaleMax: 4.5,
  scalePenalty: 0.7,

  instantConfirm: true,
  requiredConsecutive: 4,
  requiredConsecutiveBoosted: 2,
  persistIoU: 0.2,
  clearMisses: 5,

  overlay: {
    detectionBoxes: true,
    faceMesh: true,
    faceContours: true,
    pose: true,
    hands: true,
    zones: true,
    labelBanner: true,
  },

  enableHands: true,
  enablePose: true,

  debug: false,
}

/** Merge partial overrides onto the defaults (shallow, plus nested overlay). */
export function resolveConfig(
  overrides?: Partial<PhonePipelineConfig>,
): PhonePipelineConfig {
  if (!overrides) return DEFAULT_PHONE_CONFIG
  return {
    ...DEFAULT_PHONE_CONFIG,
    ...overrides,
    overlay: { ...DEFAULT_PHONE_CONFIG.overlay, ...(overrides.overlay ?? {}) },
  }
}
