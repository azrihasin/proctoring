// Shared types for the combined object-detector + MediaPipe phone-usage
// confirmation pipeline. Kept dependency-free so every module (detector, feature
// extractor, zone calc, rule engine, overlay, evidence recorder, hook) can
// import from here without pulling in React or MediaPipe.

// ---------------------------------------------------------------------------
// Geometry — everything downstream of the object detector works in NORMALIZED
// video coordinates (0..1), so thresholds are resolution-independent.
// ---------------------------------------------------------------------------
export interface NormBox {
  x: number // left, 0..1
  y: number // top, 0..1
  width: number // 0..1
  height: number // 0..1
}

export interface NormPoint {
  x: number
  y: number
  z?: number
}

// ---------------------------------------------------------------------------
// Object-detector output
// ---------------------------------------------------------------------------
export interface Detection {
  classId: number
  className: string
  score: number
  box: NormBox
}

export interface DetectionFrame {
  /** Highest-confidence person box, if any (used for spatial association). */
  person: Detection | null
  /** All person detections. */
  persons: Detection[]
  /** Every "cell phone" detection — a CANDIDATE only, never a confirmation. */
  phoneCandidates: Detection[]
  /** Other kept detections (confusers/objects) purely for the overlay. */
  objects: Detection[]
}

// ---------------------------------------------------------------------------
// MediaPipe-derived features
// ---------------------------------------------------------------------------
export interface HandFeature {
  /** 21 hand landmarks, normalized. */
  landmarks: NormPoint[]
  /** Landmark 0. */
  wrist: NormPoint
  /** Mean of palm-base landmarks (0,5,9,13,17). */
  palmCenter: NormPoint
  /** Fingertip landmarks (4,8,12,16,20). */
  fingertips: NormPoint[]
  /** Tight bbox around the hand. */
  box: NormBox
  handedness: 'Left' | 'Right' | 'Unknown'
}

export interface PoseFeature {
  leftShoulder: NormPoint | null
  rightShoulder: NormPoint | null
  leftElbow: NormPoint | null
  rightElbow: NormPoint | null
  leftWrist: NormPoint | null
  rightWrist: NormPoint | null
  leftHip: NormPoint | null
  rightHip: NormPoint | null
  /** Mean shoulder y — the "chest line" reference (0..1, top=0). */
  chestLineY: number | null
  /** Mean hip y — the lower-torso reference. */
  hipLineY: number | null
  /** Raw 33 landmarks for overlay. */
  landmarks: NormPoint[]
}

export interface HeadFeature {
  present: boolean
  /** Pitch in degrees; positive = looking DOWN. */
  pitchDeg: number
  /** Yaw in degrees; magnitude = turned away from camera. */
  yawDeg: number
  /** Gaze-down signal from eye blendshapes (0..1). */
  gazeDown: number
  /** Looking meaningfully downward (head pitch OR gaze). */
  headDown: boolean
  /** Turned away from facing the camera. */
  turnedAway: boolean
  /** Approx nose-tip position, normalized, for face-orientation checks. */
  noseTip: NormPoint | null
  /** 468 face-mesh landmarks for overlay. */
  landmarks: NormPoint[]
}

export interface MediaPipeFeatures {
  head: HeadFeature
  pose: PoseFeature | null
  hands: HandFeature[]
}

// ---------------------------------------------------------------------------
// Rule engine
// ---------------------------------------------------------------------------
export type PhoneUsageLabel =
  | 'NO_VIOLATION'
  | 'PHONE_CANDIDATE_REJECTED'
  | 'SUSPICIOUS_PHONE_CANDIDATE'
  | 'CONFIRMED_PHONE_USAGE'

/** The four required evidence gates for a confirmed phone-usage violation. */
export interface GateResults {
  /** 1. Candidate spatially associated with the student (person box). */
  personAssociation: boolean
  /** 2. Candidate near/overlapping a hand (wrist, palm, fingers, between hands). */
  handInteraction: boolean
  /** 3. Face/head/gaze supports phone usage (down / toward candidate). */
  headGaze: boolean
  /** 4. Candidate or interacting hand in a suspicious webcam zone. */
  suspiciousZone: boolean
}

export interface PhoneUsageDecision {
  label: PhoneUsageLabel
  /** The candidate this decision is about (best candidate), if any. */
  candidate: Detection | null
  gates: GateResults
  /** Names of gates that passed, e.g. ['personAssociation','handInteraction']. */
  passedGates: string[]
  /** Human-readable, explainable reasons (positive and rejecting). */
  reasons: string[]
  /** Which hand (index into features.hands) interacts with the candidate, or -1. */
  interactingHandIndex: number
  /**
   * Final combined confidence from the stateless verifier:
   *   handOverlapGate(0|1) × rawDetectorScore × (gripScore⊕geometryScore).
   * Only a candidate whose value clears cfg.displayThreshold is boxed + recorded.
   */
  combinedConfidence: number
  /** Grip-pose score (0..1) of the interacting hand, if any. */
  gripScore: number
  /** Box-geometry score (0..1): aspect band × optional face-scale sanity. */
  geometryScore: number
}

// ---------------------------------------------------------------------------
// Evidence
// ---------------------------------------------------------------------------
export interface PhoneEvidenceMeta {
  timestamp: string // ISO
  sessionId: string | null
  studentId: string | null
  label: PhoneUsageLabel
  candidate: { box: NormBox; score: number } | null
  gates: GateResults
  passedGates: string[]
  reasons: string[]
  features: {
    headPitchDeg: number
    headYawDeg: number
    gazeDown: number
    headDown: boolean
    handCount: number
    interactingHandIndex: number
    chestLineY: number | null
    hipLineY: number | null
  }
}

export interface PhoneEvidence {
  meta: PhoneEvidenceMeta
  /** Raw video frame (no overlays). */
  rawFrame?: Blob
  /** Annotated frame (detector boxes + face mesh + pose + hands + zones). */
  annotatedFrame?: Blob
}
