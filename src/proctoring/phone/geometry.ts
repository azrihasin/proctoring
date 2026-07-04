// Small normalized-coordinate geometry helpers shared by the zone calculator
// and rule engine. All boxes/points are in 0..1 video-normalized space.
import type { NormBox, NormPoint } from './types'

export function boxCenter(b: NormBox): NormPoint {
  return { x: b.x + b.width / 2, y: b.y + b.height / 2 }
}

export function boxIoU(a: NormBox, b: NormBox): number {
  const x1 = Math.max(a.x, b.x)
  const y1 = Math.max(a.y, b.y)
  const x2 = Math.min(a.x + a.width, b.x + b.width)
  const y2 = Math.min(a.y + a.height, b.y + b.height)
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  if (inter <= 0) return 0
  return inter / (a.width * a.height + b.width * b.height - inter)
}

export function dist(a: NormPoint, b: NormPoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

/** True if point p is inside box b (optionally expanded by margin on all sides). */
export function pointInBox(p: NormPoint, b: NormBox, margin = 0): boolean {
  return (
    p.x >= b.x - margin &&
    p.x <= b.x + b.width + margin &&
    p.y >= b.y - margin &&
    p.y <= b.y + b.height + margin
  )
}

export function boxArea(b: NormBox): number {
  return b.width * b.height
}

/**
 * Expand a box by `margin` (fraction of its OWN width/height) on every side,
 * clamped to the 0..1 frame. Used to grow a hand's tight bbox into the region a
 * held phone can overlap (the task's ~30–40% hand margin).
 */
export function expandBox(b: NormBox, margin: number): NormBox {
  const mx = b.width * margin
  const my = b.height * margin
  const x = Math.max(0, b.x - mx)
  const y = Math.max(0, b.y - my)
  const right = Math.min(1, b.x + b.width + mx)
  const bottom = Math.min(1, b.y + b.height + my)
  return { x, y, width: Math.max(0, right - x), height: Math.max(0, bottom - y) }
}

/** Area of the intersection of two boxes (0 if disjoint). */
export function intersectionArea(a: NormBox, b: NormBox): number {
  const x1 = Math.max(a.x, b.x)
  const y1 = Math.max(a.y, b.y)
  const x2 = Math.min(a.x + a.width, b.x + b.width)
  const y2 = Math.min(a.y + a.height, b.y + b.height)
  return Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
}

/** Bounding box of a set of points. */
export function pointsBox(points: NormPoint[]): NormBox {
  let minX = 1, minY = 1, maxX = 0, maxY = 0
  for (const p of points) {
    if (p.x < minX) minX = p.x
    if (p.y < minY) minY = p.y
    if (p.x > maxX) maxX = p.x
    if (p.y > maxY) maxY = p.y
  }
  return { x: minX, y: minY, width: Math.max(0, maxX - minX), height: Math.max(0, maxY - minY) }
}

export function mean(points: (NormPoint | null)[]): NormPoint | null {
  const valid = points.filter((p): p is NormPoint => p != null)
  if (!valid.length) return null
  const sx = valid.reduce((s, p) => s + p.x, 0)
  const sy = valid.reduce((s, p) => s + p.y, 0)
  return { x: sx / valid.length, y: sy / valid.length }
}
