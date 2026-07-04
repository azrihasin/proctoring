// Webcam-zone calculator — converts the person box + pose torso lines into the
// "suspicious lower-frame" geometry the rule engine (gate 4) and overlay use.
//
// A phone held for use during an exam is almost always LOW: below chest level,
// in the lower torso, in the bottom third of the person box, or near the bottom
// edge of the frame. This module turns those descriptions into concrete
// normalized y-thresholds and rectangles. All thresholds come from config, so
// the "suspicious zone" can be retuned without touching the rules.

import type { PhonePipelineConfig } from './config'
import type { NormBox, NormPoint, PoseFeature, Detection } from './types'
import { boxCenter } from './geometry'

export interface ZoneRect extends NormBox {
  label: string
}

export interface WebcamZones {
  /** y below which is "below chest level" (from pose shoulders, else fallback). */
  chestLineY: number
  /** y below which is "lower torso" (from pose hips, else derived). */
  lowerTorsoY: number
  /** y below which is "bottom N% of the person box". */
  bottomPersonY: number
  /** y below which is "near the bottom edge of the frame". */
  bottomFrameY: number
  /** The single highest (smallest y) threshold — anything below this is suspicious. */
  suspiciousAboveY: number
  /** Rectangles for the overlay. */
  rects: ZoneRect[]
  /** True if a point is in any suspicious zone. */
  isPointSuspicious(p: NormPoint): boolean
  /** True if a box's center or lower edge falls in a suspicious zone. */
  isBoxSuspicious(b: NormBox): boolean
}

export function computeWebcamZones(
  cfg: PhonePipelineConfig,
  person: Detection | null,
  pose: PoseFeature | null,
): WebcamZones {
  // Chest line: pose shoulders if available, else 40% down the person box, else
  // 45% down the frame.
  const chestLineY =
    pose?.chestLineY ??
    (person ? person.box.y + person.box.height * 0.4 : 0.45)

  // Lower torso: pose hips if available, else 75% down the person box, else 0.7.
  const lowerTorsoY =
    pose?.hipLineY ??
    (person ? person.box.y + person.box.height * 0.75 : 0.7)

  // Bottom N% of the person box.
  const bottomPersonY = person
    ? person.box.y + person.box.height * (1 - cfg.bottomPersonZone)
    : 1 - cfg.bottomPersonZone

  // Bottom N% of the frame.
  const bottomFrameY = 1 - cfg.bottomFrameZone

  // The union of all suspicious zones = everything below the SHALLOWEST line.
  const suspiciousAboveY = Math.min(chestLineY, lowerTorsoY, bottomPersonY, bottomFrameY)

  const rects: ZoneRect[] = [
    { label: 'below chest', x: 0, y: chestLineY, width: 1, height: Math.max(0, 1 - chestLineY) },
    { label: 'lower torso', x: 0, y: lowerTorsoY, width: 1, height: Math.max(0, 1 - lowerTorsoY) },
    { label: 'bottom frame', x: 0, y: bottomFrameY, width: 1, height: Math.max(0, 1 - bottomFrameY) },
  ]
  if (person) {
    rects.push({
      label: 'lower person box',
      x: person.box.x,
      y: bottomPersonY,
      width: person.box.width,
      height: Math.max(0, person.box.y + person.box.height - bottomPersonY),
    })
  }

  const isPointSuspicious = (p: NormPoint) => p.y >= suspiciousAboveY

  const isBoxSuspicious = (b: NormBox) => {
    const c = boxCenter(b)
    const lowerEdge = b.y + b.height
    return c.y >= suspiciousAboveY || lowerEdge >= suspiciousAboveY
  }

  return {
    chestLineY,
    lowerTorsoY,
    bottomPersonY,
    bottomFrameY,
    suspiciousAboveY,
    rects,
    isPointSuspicious,
    isBoxSuspicious,
  }
}
