// Evidence recorder — for a CONFIRMED_PHONE_USAGE it captures an explainable
// bundle: the raw frame, an annotated frame (detector boxes + Face Mesh + pose +
// hands + zones), and a JSON metadata record (timestamp, session/student id,
// candidate box + confidence, head/hand/gaze features, the gates that passed,
// and the reason list). Kept independent of the detection loop so what/where it
// saves can change without touching the pipeline.

import type { PhonePipelineConfig } from './config'
import type {
  PhoneEvidence,
  PhoneEvidenceMeta,
  PhoneUsageDecision,
} from './types'
import type { OverlaySnapshot } from './overlay'
import { renderPhoneOverlay } from './overlay'

export interface EvidenceIds {
  sessionId: string | null
  studentId: string | null
}

function toPngBlob(canvas: HTMLCanvasElement): Promise<Blob | undefined> {
  return new Promise((resolve) => {
    canvas.toBlob((b) => resolve(b ?? undefined), 'image/png')
  })
}

function buildMeta(
  decision: PhoneUsageDecision,
  snap: OverlaySnapshot,
  ids: EvidenceIds,
): PhoneEvidenceMeta {
  const { head, pose } = snap.features
  return {
    timestamp: new Date().toISOString(),
    sessionId: ids.sessionId,
    studentId: ids.studentId,
    label: decision.label,
    candidate: decision.candidate
      ? { box: decision.candidate.box, score: decision.candidate.score }
      : null,
    gates: decision.gates,
    passedGates: decision.passedGates,
    reasons: decision.reasons,
    features: {
      headPitchDeg: head.pitchDeg,
      headYawDeg: head.yawDeg,
      gazeDown: head.gazeDown,
      headDown: head.headDown,
      handCount: snap.features.hands.length,
      interactingHandIndex: decision.interactingHandIndex,
      chestLineY: pose?.chestLineY ?? null,
      hipLineY: pose?.hipLineY ?? null,
    },
  }
}

/**
 * Capture the raw + annotated frames and metadata for a decision. Uses offscreen
 * canvases sized to the video so the annotated frame matches the live overlay.
 */
export async function captureEvidence(
  video: HTMLVideoElement,
  cfg: PhonePipelineConfig,
  snap: OverlaySnapshot,
  ids: EvidenceIds,
): Promise<PhoneEvidence> {
  const w = video.videoWidth
  const h = video.videoHeight
  const meta = buildMeta(snap.decision, snap, ids)

  if (!w || !h) return { meta }

  // Raw frame.
  const rawCanvas = document.createElement('canvas')
  rawCanvas.width = w
  rawCanvas.height = h
  const rawCtx = rawCanvas.getContext('2d')
  let rawFrame: Blob | undefined
  if (rawCtx) {
    rawCtx.drawImage(video, 0, 0, w, h)
    rawFrame = await toPngBlob(rawCanvas)
  }

  // Annotated frame = raw frame + all overlays.
  const annCanvas = document.createElement('canvas')
  annCanvas.width = w
  annCanvas.height = h
  const annCtx = annCanvas.getContext('2d')
  let annotatedFrame: Blob | undefined
  if (annCtx) {
    annCtx.drawImage(video, 0, 0, w, h)
    renderPhoneOverlay(annCtx, cfg, snap)
    annotatedFrame = await toPngBlob(annCanvas)
  }

  return { meta, rawFrame, annotatedFrame }
}

/** Convenience: trigger a browser download of the whole evidence bundle. */
export function downloadEvidence(evidence: PhoneEvidence): void {
  const stamp = evidence.meta.timestamp.replace(/[:.]/g, '-')
  const base = `phone_evidence_${stamp}`

  const saveBlob = (blob: Blob | undefined, name: string) => {
    if (!blob) return
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = name
    a.style.display = 'none'
    document.body.appendChild(a)
    a.click()
    setTimeout(() => {
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    }, 100)
  }

  saveBlob(evidence.rawFrame, `${base}_raw.png`)
  saveBlob(evidence.annotatedFrame, `${base}_annotated.png`)
  saveBlob(
    new Blob([JSON.stringify(evidence.meta, null, 2)], { type: 'application/json' }),
    `${base}_meta.json`,
  )
}
