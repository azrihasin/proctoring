// EfficientDet-Lite2 COCO detector (MediaPipe Tasks Vision ObjectDetector) — the
// phone-candidate detector. It produces a per-frame DetectionFrame the rule
// engine, zone calculator, overlay and hook consume:
//   • person boxes        (COCO "person")     → spatial association for the student
//   • phone CANDIDATES     (COCO "cell phone") → never a confirmation on their own
//   • object boxes         (confuser set)      → drawn on the overlay for context
//
// EfficientDet-Lite2 is a stock 80-class COCO detector, so it exposes
// "cell phone" / "person" / look-alike classes directly by NAME. All candidate
// rejection (person association, hand interaction, head/gaze, zone) is still the
// rule engine's job; this file only surfaces PERMISSIVE candidates.
//
// It runs on the MediaPipe WASM runtime already loaded for face/pose/hands, so
// there is no separate model runtime to download.

import { FilesetResolver, ObjectDetector } from '@mediapipe/tasks-vision'
import type { NormBox, Detection, DetectionFrame } from './types'

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'

// COCO class NAMES (EfficientDet-Lite label map) we care about. We match by name
// rather than index because MediaPipe surfaces category names, not COCO ids.
const PERSON_NAME = 'person'
const CELL_PHONE_NAME = 'cell phone'
// Look-alike classes worth drawing so a reviewer can see WHY a candidate near a
// "book"/"laptop"/"chair" was rejected:
// tv · laptop · mouse · remote · keyboard · book · chair · couch · bed ·
// dining table.
const OBJECT_NAMES = new Set([
  'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'book',
  'chair', 'couch', 'bed', 'dining table',
])

export interface EfficientDetCandidateOptions {
  modelUrl?: string
  candidateScoreThreshold?: number
  personScoreThreshold?: number
  objectScoreThreshold?: number
  /** Max detections MediaPipe returns per frame before our per-class filtering. */
  maxResults?: number
}

const DEFAULTS = {
  // EfficientDet-Lite2 (float16) COCO object detector, same mediapipe-models
  // host the face/pose/hand tasks are pulled from.
  modelUrl:
    'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/1/efficientdet_lite2.tflite',
  candidateScoreThreshold: 0.4,
  personScoreThreshold: 0.5,
  objectScoreThreshold: 0.45,
  maxResults: 25,
}

export class EfficientDetCandidateDetector {
  private detector: ObjectDetector
  private opts: Required<EfficientDetCandidateOptions>
  /** Monotonic timestamp for detectForVideo (its own stream, ms). */
  private lastTs = 0
  readonly backend: 'GPU' | 'CPU'

  private constructor(
    detector: ObjectDetector,
    opts: Required<EfficientDetCandidateOptions>,
    backend: 'GPU' | 'CPU',
  ) {
    this.detector = detector
    this.opts = opts
    this.backend = backend
  }

  static async create(
    options: EfficientDetCandidateOptions = {},
  ): Promise<EfficientDetCandidateDetector> {
    const opts = { ...DEFAULTS, ...options } as Required<EfficientDetCandidateOptions>
    const vision = await FilesetResolver.forVisionTasks(WASM_URL)

    // The detector's own score gate must sit at the LOWEST of our per-class
    // thresholds so nothing we later want is dropped inside MediaPipe.
    const floor = Math.min(
      opts.candidateScoreThreshold,
      opts.personScoreThreshold,
      opts.objectScoreThreshold,
    )

    const make = (delegate: 'GPU' | 'CPU') =>
      ObjectDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: opts.modelUrl, delegate },
        runningMode: 'VIDEO',
        scoreThreshold: floor,
        maxResults: opts.maxResults,
      })

    let backend: 'GPU' | 'CPU' = 'GPU'
    let detector: ObjectDetector
    try {
      detector = await make('GPU')
    } catch (gpuErr) {
      console.warn('⚠️ EfficientDet GPU delegate failed, falling back to CPU', gpuErr)
      detector = await make('CPU')
      backend = 'CPU'
    }
    console.debug(`[phone] EfficientDet-Lite2 candidate detector ready on "${backend}"`)
    return new EfficientDetCandidateDetector(detector, opts, backend)
  }

  /** Run one frame; returns persons, phone candidates and objects (normalized). */
  async detect(video: HTMLVideoElement): Promise<DetectionFrame> {
    const srcW = video.videoWidth
    const srcH = video.videoHeight
    if (!srcW || !srcH) return { person: null, persons: [], phoneCandidates: [], objects: [] }

    // ObjectDetector VIDEO mode requires strictly increasing timestamps.
    let ts = performance.now()
    if (ts <= this.lastTs) ts = this.lastTs + 1
    this.lastTs = ts

    const result = this.detector.detectForVideo(video, ts)

    const persons: Detection[] = []
    const phoneCandidates: Detection[] = []
    const objects: Detection[] = []

    for (const d of result.detections) {
      const cat = d.categories?.[0]
      const bbox = d.boundingBox
      if (!cat || !bbox) continue
      const name = (cat.categoryName || '').toLowerCase()
      const score = cat.score
      const det: Detection = {
        classId: -1, // COCO id unused downstream; keep name-based identity.
        className: name,
        score,
        box: pxToNorm(bbox, srcW, srcH),
      }

      if (name === PERSON_NAME) {
        if (score >= this.opts.personScoreThreshold) persons.push(det)
      } else if (name === CELL_PHONE_NAME) {
        if (score >= this.opts.candidateScoreThreshold) phoneCandidates.push(det)
      } else if (OBJECT_NAMES.has(name) && score >= this.opts.objectScoreThreshold) {
        objects.push(det)
      }
    }

    persons.sort((a, b) => b.score - a.score)
    phoneCandidates.sort((a, b) => b.score - a.score)

    return {
      person: persons[0] ?? null,
      persons,
      phoneCandidates,
      objects,
    }
  }

  async close(): Promise<void> {
    this.detector.close()
  }
}

/** MediaPipe pixel bbox → normalized (0..1) box, clamped to the frame. */
function pxToNorm(
  bbox: { originX: number; originY: number; width: number; height: number },
  srcW: number,
  srcH: number,
): NormBox {
  const x = Math.max(0, bbox.originX)
  const y = Math.max(0, bbox.originY)
  const right = Math.min(srcW, bbox.originX + bbox.width)
  const bottom = Math.min(srcH, bbox.originY + bbox.height)
  return {
    x: x / srcW,
    y: y / srcH,
    width: Math.max(0, right - x) / srcW,
    height: Math.max(0, bottom - y) / srcH,
  }
}
