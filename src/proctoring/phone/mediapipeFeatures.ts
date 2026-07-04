// MediaPipe feature extractor — the single place that runs FaceLandmarker,
// PoseLandmarker and HandLandmarker and turns their raw output into the compact
// feature set the rule engine reasons about (head/gaze, shoulders/wrists/torso,
// hand landmarks). Overlay data (raw landmark arrays) is passed through so the
// renderer can draw the mesh / skeleton / hands.
//
// Perception thresholds (what counts as "head down", "turned away") live here
// and are read from the shared config, so they can be tuned without touching the
// confirmation/rejection rules in ruleEngine.ts.

import {
  FilesetResolver,
  FaceLandmarker,
  PoseLandmarker,
  HandLandmarker,
  type FaceLandmarkerResult,
  type PoseLandmarkerResult,
  type HandLandmarkerResult,
} from '@mediapipe/tasks-vision'
import type { PhonePipelineConfig } from './config'
import type {
  HandFeature,
  HeadFeature,
  MediaPipeFeatures,
  NormPoint,
  PoseFeature,
} from './types'
import { mean, pointsBox } from './geometry'

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
const FACE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
const POSE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
const HAND_MODEL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

// ---- Face-mesh landmark indices ----
const NOSE_TIP = 1
const FOREHEAD = 10
const CHIN = 152
const LEFT_CHEEK = 234
const RIGHT_CHEEK = 454

// ---- Pose landmark indices ----
const P_LEFT_SHOULDER = 11
const P_RIGHT_SHOULDER = 12
const P_LEFT_ELBOW = 13
const P_RIGHT_ELBOW = 14
const P_LEFT_WRIST = 15
const P_RIGHT_WRIST = 16
const P_LEFT_HIP = 23
const P_RIGHT_HIP = 24
const POSE_MIN_VISIBILITY = 0.3

// ---- Hand landmark indices ----
const PALM_BASE = [0, 5, 9, 13, 17]
const FINGERTIPS = [4, 8, 12, 16, 20]

export interface RawMediaPipeResults {
  face: FaceLandmarkerResult | null
  pose: PoseLandmarkerResult | null
  hands: HandLandmarkerResult | null
}

export class MediaPipeFeatureExtractor {
  private cfg: PhonePipelineConfig
  private faceLm: FaceLandmarker | null
  private poseLm: PoseLandmarker | null
  private handLm: HandLandmarker | null

  private constructor(
    cfg: PhonePipelineConfig,
    faceLm: FaceLandmarker | null,
    poseLm: PoseLandmarker | null,
    handLm: HandLandmarker | null,
  ) {
    this.cfg = cfg
    this.faceLm = faceLm
    this.poseLm = poseLm
    this.handLm = handLm
  }

  get hasFace() { return this.faceLm !== null }
  get hasPose() { return this.poseLm !== null }
  get hasHands() { return this.handLm !== null }

  static async create(cfg: PhonePipelineConfig): Promise<MediaPipeFeatureExtractor> {
    const vision = await FilesetResolver.forVisionTasks(WASM_URL)

    const withFallback = async <T>(make: (d: 'GPU' | 'CPU') => Promise<T>, label: string) => {
      try { return await make('GPU') }
      catch (e) {
        console.warn(`⚠️ ${label} GPU delegate failed, falling back to CPU`, e)
        return await make('CPU')
      }
    }

    const faceLm = await withFallback(
      (delegate) => FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: FACE_MODEL, delegate },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      }),
      'FaceLandmarker',
    ).catch((e) => { console.warn('⚠️ FaceLandmarker disabled', e); return null })

    let poseLm: PoseLandmarker | null = null
    if (cfg.enablePose) {
      poseLm = await withFallback(
        (delegate) => PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: POSE_MODEL, delegate },
          runningMode: 'VIDEO',
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        }),
        'PoseLandmarker',
      ).catch((e) => { console.warn('⚠️ PoseLandmarker disabled', e); return null })
    }

    let handLm: HandLandmarker | null = null
    if (cfg.enableHands) {
      handLm = await withFallback(
        (delegate) => HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: HAND_MODEL, delegate },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.4,
          minHandPresenceConfidence: 0.4,
          minTrackingConfidence: 0.4,
        }),
        'HandLandmarker',
      ).catch((e) => { console.warn('⚠️ HandLandmarker disabled', e); return null })
    }

    return new MediaPipeFeatureExtractor(cfg, faceLm, poseLm, handLm)
  }

  /** Run all enabled MediaPipe models for one frame at monotonic timestamp ts. */
  detect(video: HTMLVideoElement, ts: number): RawMediaPipeResults {
    const face = this.faceLm?.detectForVideo(video, ts) ?? null
    const pose = this.poseLm?.detectForVideo(video, ts) ?? null
    const hands = this.handLm?.detectForVideo(video, ts) ?? null
    return { face, pose, hands }
  }

  /** Convert raw MediaPipe output into the compact feature set. */
  extract(raw: RawMediaPipeResults): MediaPipeFeatures {
    return {
      head: this.extractHead(raw.face),
      pose: this.extractPose(raw.pose),
      hands: this.extractHands(raw.hands),
    }
  }

  // -------------------------------------------------------------------------
  private extractHead(face: FaceLandmarkerResult | null): HeadFeature {
    const empty: HeadFeature = {
      present: false, pitchDeg: 0, yawDeg: 0, gazeDown: 0,
      headDown: false, turnedAway: false, noseTip: null, landmarks: [],
    }
    if (!face || !face.faceLandmarks?.length) return empty
    const lms = face.faceLandmarks[0] as NormPoint[]
    if (lms.length < 468) return empty

    // --- Euler angles from the facial transformation matrix (for evidence) ---
    let pitchDeg = 0, yawDeg = 0
    const mat = face.facialTransformationMatrixes?.[0]?.data
    if (mat && mat.length >= 12) {
      // Column-major 4x4. Rotation upper-left 3x3.
      const r20 = mat[2], r21 = mat[6], r22 = mat[10]
      const r10 = mat[1], r00 = mat[0]
      pitchDeg = (Math.atan2(r21, r22) * 180) / Math.PI
      yawDeg = (Math.atan2(-r20, Math.hypot(r21, r22)) * 180) / Math.PI
      void r10; void r00
    }

    // --- Gaze-down from eye blendshapes (unambiguous sign) ---
    let gazeDown = 0
    const shapes = face.faceBlendshapes?.[0]?.categories
    if (shapes) {
      const down = shapes.filter((c) =>
        c.categoryName === 'eyeLookDownLeft' || c.categoryName === 'eyeLookDownRight')
      if (down.length) gazeDown = down.reduce((s, c) => s + c.score, 0) / down.length
    }

    // --- Landmark heuristics (robust, sign-stable backups) ---
    const noseTip = lms[NOSE_TIP]
    const forehead = lms[FOREHEAD]
    const chin = lms[CHIN]
    const leftCheek = lms[LEFT_CHEEK]
    const rightCheek = lms[RIGHT_CHEEK]

    // Yaw magnitude: nose-tip x offset from the cheekbone midpoint.
    const cheekMidX = (leftCheek.x + rightCheek.x) / 2
    const cheekSpan = Math.max(Math.abs(rightCheek.x - leftCheek.x), 0.05)
    const yawOffset = Math.abs(noseTip.x - cheekMidX) / cheekSpan // ~0 forward, grows turning away

    // Pitch-down proxy: chin comes forward (smaller z) relative to forehead.
    const zDown = (chin.z ?? 0) - (forehead.z ?? 0) // negative → looking down

    const headDown =
      gazeDown >= this.cfg.gazeDownThreshold ||
      zDown < -0.05 ||
      (pitchDeg >= this.cfg.headPitchDownDeg) // matrix pitch, if its sign agrees

    const turnedAway =
      yawOffset > 0.22 || Math.abs(yawDeg) >= this.cfg.headYawAwayDeg

    return {
      present: true,
      pitchDeg,
      yawDeg,
      gazeDown,
      headDown,
      turnedAway,
      noseTip: { x: noseTip.x, y: noseTip.y },
      landmarks: lms,
    }
  }

  // -------------------------------------------------------------------------
  private extractPose(pose: PoseLandmarkerResult | null): PoseFeature | null {
    if (!pose || !pose.landmarks?.length) return null
    const lms = pose.landmarks[0] as Array<NormPoint & { visibility?: number }>

    const pick = (i: number): NormPoint | null => {
      const p = lms[i]
      if (!p) return null
      if ((p.visibility ?? 1) < POSE_MIN_VISIBILITY) return null
      return { x: p.x, y: p.y, z: p.z }
    }

    const leftShoulder = pick(P_LEFT_SHOULDER)
    const rightShoulder = pick(P_RIGHT_SHOULDER)
    const leftHip = pick(P_LEFT_HIP)
    const rightHip = pick(P_RIGHT_HIP)

    const shoulderMid = mean([leftShoulder, rightShoulder])
    const hipMid = mean([leftHip, rightHip])

    return {
      leftShoulder,
      rightShoulder,
      leftElbow: pick(P_LEFT_ELBOW),
      rightElbow: pick(P_RIGHT_ELBOW),
      leftWrist: pick(P_LEFT_WRIST),
      rightWrist: pick(P_RIGHT_WRIST),
      leftHip,
      rightHip,
      chestLineY: shoulderMid ? shoulderMid.y : null,
      hipLineY: hipMid ? hipMid.y : null,
      landmarks: lms.map((p) => ({ x: p.x, y: p.y, z: p.z })),
    }
  }

  // -------------------------------------------------------------------------
  private extractHands(hands: HandLandmarkerResult | null): HandFeature[] {
    if (!hands || !hands.landmarks?.length) return []
    const out: HandFeature[] = []
    hands.landmarks.forEach((lmSet, i) => {
      const landmarks = (lmSet as NormPoint[]).map((p) => ({ x: p.x, y: p.y, z: p.z }))
      if (landmarks.length < 21) return
      const wrist = landmarks[0]
      const palmCenter = mean(PALM_BASE.map((idx) => landmarks[idx])) ?? wrist
      const fingertips = FINGERTIPS.map((idx) => landmarks[idx])
      const handednessCat = hands.handedness?.[i]?.[0]?.categoryName
      const handedness: HandFeature['handedness'] =
        handednessCat === 'Left' ? 'Left' : handednessCat === 'Right' ? 'Right' : 'Unknown'
      out.push({
        landmarks,
        wrist,
        palmCenter,
        fingertips,
        box: pointsBox(landmarks),
        handedness,
      })
    })
    return out
  }

  close(): void {
    this.faceLm?.close()
    this.poseLm?.close()
    this.handLm?.close()
  }
}
