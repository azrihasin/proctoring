// Overlay / visualization renderer — draws every utility overlay onto a 2D
// canvas context so both the LIVE preview and the SAVED evidence frame show the
// same explainable annotations:
//   • Detector boxes (person / phone candidate / objects) with class + confidence
//   • MediaPipe Face Mesh tesselation + contours (gaze / head-pose reasoning)
//   • MediaPipe Pose skeleton + keypoints (shoulders / elbows / wrists / torso)
//   • MediaPipe hand landmarks (when hands are enabled)
//   • Suspicious webcam zones
//   • A decision banner (NO_VIOLATION / REJECTED / SUSPICIOUS / CONFIRMED)
//
// Individual layers are toggled via config.overlay so visibility can be tuned
// without touching detection logic. The context is expected to be sized to the
// video's pixel dimensions (normalized landmarks scale by canvas width/height).

import {
  DrawingUtils,
  FaceLandmarker,
  PoseLandmarker,
  HandLandmarker,
} from '@mediapipe/tasks-vision'
import type { PhonePipelineConfig } from './config'
import type {
  MediaPipeFeatures,
  PhoneUsageDecision,
  Detection,
  DetectionFrame,
} from './types'
import type { WebcamZones } from './webcamZones'

export interface OverlaySnapshot {
  detections: DetectionFrame
  features: MediaPipeFeatures
  zones: WebcamZones
  decision: PhoneUsageDecision
}

const LABEL_COLORS: Record<PhoneUsageDecision['label'], string> = {
  NO_VIOLATION: '#22c55e',
  PHONE_CANDIDATE_REJECTED: '#94a3b8',
  SUSPICIOUS_PHONE_CANDIDATE: '#f59e0b',
  CONFIRMED_PHONE_USAGE: '#ef4444',
}

export function renderPhoneOverlay(
  ctx: CanvasRenderingContext2D,
  cfg: PhonePipelineConfig,
  snap: OverlaySnapshot,
): void {
  const W = ctx.canvas.width
  const H = ctx.canvas.height
  const du = new DrawingUtils(ctx)
  const { detections, features, zones, decision } = snap
  const o = cfg.overlay

  // ---- Suspicious zones (drawn first, under everything) ----
  if (o.zones) drawZones(ctx, zones, W, H)

  // ---- MediaPipe Face Mesh ----
  if (o.faceMesh && features.head.landmarks.length) {
    du.drawConnectors(
      features.head.landmarks as any,
      FaceLandmarker.FACE_LANDMARKS_TESSELATION,
      { color: 'rgba(180,220,255,0.35)', lineWidth: 0.5 },
    )
    if (o.faceContours) {
      du.drawConnectors(
        features.head.landmarks as any,
        FaceLandmarker.FACE_LANDMARKS_CONTOURS,
        { color: 'rgba(120,200,255,0.9)', lineWidth: 1 },
      )
    }
  }

  // ---- MediaPipe Pose skeleton ----
  if (o.pose && features.pose && features.pose.landmarks.length) {
    du.drawConnectors(
      features.pose.landmarks as any,
      PoseLandmarker.POSE_CONNECTIONS,
      { color: 'rgba(0,255,200,0.9)', lineWidth: 2 },
    )
    du.drawLandmarks(features.pose.landmarks as any, {
      color: '#00e5ff',
      radius: 2,
    })
  }

  // ---- MediaPipe hands ----
  if (o.hands) {
    for (const hand of features.hands) {
      du.drawConnectors(hand.landmarks as any, HandLandmarker.HAND_CONNECTIONS, {
        color: 'rgba(255,120,220,0.9)',
        lineWidth: 2,
      })
      du.drawLandmarks(hand.landmarks as any, { color: '#ff4fd8', radius: 2 })
    }
  }

  // ---- Detector boxes ----
  if (o.detectionBoxes) {
    for (const p of detections.persons) drawBox(ctx, p, W, H, '#22c55e')
    for (const obj of detections.objects) drawBox(ctx, obj, W, H, '#facc15')

    // Phone candidates are only boxed once they clear the stateless verifier
    // (hand-overlap gate + grip + geometry ≥ display threshold). Raw detector
    // output that fails a gate or falls below threshold is NEVER boxed — even
    // when the model was confident — which is the whole point of the fix.
    if (decision.label === 'CONFIRMED_PHONE_USAGE' && decision.candidate) {
      drawBox(ctx, decision.candidate, W, H, LABEL_COLORS.CONFIRMED_PHONE_USAGE, 3)
    }
  }

  // ---- Decision banner ----
  if (o.labelBanner) drawBanner(ctx, decision)
}

function drawBox(
  ctx: CanvasRenderingContext2D,
  det: Detection,
  W: number,
  H: number,
  color: string,
  lineWidth = 2,
): void {
  const x = det.box.x * W
  const y = det.box.y * H
  const w = det.box.width * W
  const h = det.box.height * H
  ctx.strokeStyle = color
  ctx.lineWidth = lineWidth
  ctx.strokeRect(x, y, w, h)

  const label = `${det.className} ${(det.score * 100).toFixed(0)}%`
  ctx.font = 'bold 13px Arial'
  const tw = ctx.measureText(label).width + 8
  ctx.fillStyle = color
  ctx.fillRect(x, Math.max(0, y - 18), tw, 18)
  ctx.fillStyle = '#000'
  ctx.fillText(label, x + 4, Math.max(11, y - 5))
}

function drawZones(
  ctx: CanvasRenderingContext2D,
  zones: WebcamZones,
  W: number,
  H: number,
): void {
  // Single translucent fill for the union suspicious region, plus a dashed line
  // at the chest reference. Keeps the overlay readable rather than stacking
  // several overlapping fills.
  ctx.save()
  ctx.fillStyle = 'rgba(239,68,68,0.08)'
  const yTop = zones.suspiciousAboveY * H
  ctx.fillRect(0, yTop, W, H - yTop)

  ctx.strokeStyle = 'rgba(0,229,255,0.6)'
  ctx.setLineDash([6, 4])
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(0, zones.chestLineY * H)
  ctx.lineTo(W, zones.chestLineY * H)
  ctx.stroke()
  ctx.setLineDash([])

  ctx.fillStyle = 'rgba(0,229,255,0.9)'
  ctx.font = '11px Arial'
  ctx.fillText('chest line', 6, zones.chestLineY * H - 4)
  ctx.fillText('suspicious zone', 6, yTop + 14)
  ctx.restore()
}

function drawBanner(ctx: CanvasRenderingContext2D, decision: PhoneUsageDecision): void {
  const color = LABEL_COLORS[decision.label]
  const text = decision.label.replace(/_/g, ' ')
  ctx.save()
  ctx.font = 'bold 16px Arial'
  const w = ctx.measureText(text).width + 20
  ctx.fillStyle = color
  ctx.fillRect(10, 10, w, 28)
  ctx.fillStyle = decision.label === 'PHONE_CANDIDATE_REJECTED' ? '#0f172a' : '#ffffff'
  ctx.fillText(text, 20, 30)

  // Top reason(s) under the banner for quick live context.
  const top = decision.reasons.slice(0, 3)
  ctx.font = '12px Arial'
  top.forEach((r, i) => {
    const y = 46 + i * 16
    ctx.fillStyle = 'rgba(0,0,0,0.55)'
    const rw = ctx.measureText(r).width + 12
    ctx.fillRect(10, y - 12, rw, 16)
    ctx.fillStyle = '#fff'
    ctx.fillText(r, 16, y)
  })
  ctx.restore()
}
