import { useEffect, useRef } from 'react'
import * as faceapi from '@vladmandic/face-api'

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const MODEL_URL = '/models'
const KYC_URL = 'https://proctor-x-api.appricode.net/api/kyc/getPhoto'
const VIDEO_ELEMENT_ID = 'proctoring-video'

// face-api descriptors are compared with Euclidean distance (the metric
// FaceMatcher uses). Smaller distance = more similar. Same person is typically
// < 0.5, a different person is typically > 0.6. Cosine similarity is NOT a
// reliable metric here because the descriptors are not normalized — different
// people routinely score 0.85–0.95, so a mismatch would never be flagged.
const DISTANCE_THRESHOLD = 0.6 // above => mismatch
const GRACE_OPEN_COUNT = 5 // consecutive mismatching ticks required to open
const RESOLVE_COUNT = 3 // consecutive matching ticks required to resolve
const TICK_INTERVAL_MS = 1000 // run a comparison every 1 second

export interface FaceProctoringParams {
  sessionId: string | null
  // Called on every tick while the live face does not match the KYC selfie.
  // `distance` is the Euclidean distance at detection time (higher = less similar).
  onMismatch: (distance: number) => void
  // Called once when the live face matches again (incident resolved).
  onMatch: () => void
  // Optional: surfaces a human-readable status of the wrong-face checker
  // (disabled / KYC failure / active + live distance) so the host can show it
  // on-screen for debugging without opening the browser console.
  onStatus?: (message: string) => void
}

// ---------------------------------------------------------------------------
// Model loading (guarded so the three nets load only once per page)
// ---------------------------------------------------------------------------
let modelsPromise: Promise<void> | null = null
function loadModels(): Promise<void> {
  if (!modelsPromise) {
    modelsPromise = (async () => {
      await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL)
      await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL)
      await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL)
    })()
  }
  return modelsPromise
}

// ---------------------------------------------------------------------------
// Face-api helpers
// ---------------------------------------------------------------------------
async function extractDescriptor(
  input: HTMLImageElement | HTMLVideoElement,
): Promise<Float32Array | null> {
  const result = await faceapi
    .detectSingleFace(input, new faceapi.SsdMobilenetv1Options())
    .withFaceLandmarks()
    .withFaceDescriptor()
  return result ? result.descriptor : null
}

function euclideanDistance(a: Float32Array, b: Float32Array): number {
  // Mirrors faceapi.euclideanDistance — the metric face descriptors are
  // designed for. Lower = more similar.
  let sum = 0
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i]
    sum += diff * diff
  }
  return Math.sqrt(sum)
}

// ---------------------------------------------------------------------------
// KYC selfie -> baseline descriptor
// ---------------------------------------------------------------------------
function pickBase64(payload: unknown): string {
  if (typeof payload === 'string') return payload
  if (payload && typeof payload === 'object') {
    const obj = payload as Record<string, unknown>
    const candidate =
      obj.photo ?? obj.data ?? obj.image ?? obj.base64 ?? obj.selfie ?? obj.photoBase64
    if (typeof candidate === 'string') return candidate
  }
  return ''
}

function base64ToImage(base64: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const src = base64.startsWith('data:') ? base64 : `data:image/jpeg;base64,${base64}`
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('Failed to decode KYC selfie image'))
    img.src = src
  })
}

async function fetchKycDescriptor(sessionId: string): Promise<Float32Array | null> {
  const url = `${KYC_URL}?sessionId=${encodeURIComponent(sessionId)}&photoType=selfie`
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`KYC getPhoto failed: ${res.status} ${res.statusText}`)
  }
  // The endpoint may return a raw base64 string or a JSON object wrapping it.
  const text = await res.text()
  let base64 = text
  try {
    base64 = pickBase64(JSON.parse(text))
  } catch {
    // Not JSON — treat the body as the raw base64 string.
  }
  if (!base64) {
    throw new Error('KYC getPhoto response did not contain a base64 image')
  }
  const img = await base64ToImage(base64)
  return extractDescriptor(img)
}

// ---------------------------------------------------------------------------
// Hook
//
// Compares the live webcam feed against the KYC selfie once a second. It does
// NOT send any events or uploads itself — instead it reports mismatches through
// the onMismatch / onMatch callbacks so the host can funnel them through the
// same violation-log / event / upload pipeline as every other violation type.
// ---------------------------------------------------------------------------
export function useFaceProctoring({ sessionId, onMismatch, onMatch, onStatus }: FaceProctoringParams) {
  // State-machine bookkeeping that must survive across interval ticks.
  const kycDescriptorRef = useRef<Float32Array | null>(null)
  const openRef = useRef(false) // whether a mismatch incident is currently open
  const belowCountRef = useRef(0) // consecutive mismatching ticks
  const aboveCountRef = useRef(0) // consecutive matching ticks
  const tickRunningRef = useRef(false)

  // Keep the latest callbacks in refs so the long-lived interval always calls
  // the current versions without needing to restart the effect.
  const onMismatchRef = useRef(onMismatch)
  const onMatchRef = useRef(onMatch)
  const onStatusRef = useRef(onStatus)
  useEffect(() => {
    onMismatchRef.current = onMismatch
    onMatchRef.current = onMatch
    onStatusRef.current = onStatus
  })
  const status = (message: string) => onStatusRef.current?.(message)

  useEffect(() => {
    if (!sessionId) {
      console.warn('⚠️ useFaceProctoring: missing sessionId, proctoring disabled')
      status('Wrong-face check DISABLED: no sessionId/idNo in URL')
      return
    }

    let cancelled = false
    let intervalId: number | null = null

    const getVideo = (): HTMLVideoElement | null => {
      const el = document.getElementById(VIDEO_ELEMENT_ID) as HTMLVideoElement | null
      if (!el) return null
      if (el.readyState < 2 || el.videoWidth === 0 || el.paused) return null
      return el
    }

    const tick = async () => {
      if (tickRunningRef.current) return // avoid overlapping ticks
      tickRunningRef.current = true
      try {
        const baseline = kycDescriptorRef.current
        if (!baseline) return
        const video = getVideo()
        if (!video) return

        // MediaPipe already guarantees exactly one face is present here, so we
        // only need the descriptor — no face-count / face-absence checks.
        const descriptor = await extractDescriptor(video)
        if (!descriptor) return // descriptor occasionally unavailable; skip this tick

        const distance = euclideanDistance(descriptor, baseline)
        const isMismatch = distance > DISTANCE_THRESHOLD
        console.debug(
          `🧑‍🤝‍🧑 face distance=${distance.toFixed(3)} (threshold ${DISTANCE_THRESHOLD}) -> ${
            isMismatch ? 'MISMATCH' : 'match'
          }`,
        )
        // Surface the live reading so the host can confirm the checker is alive
        // and see how close the comparison is to the mismatch threshold.
        status(
          `Wrong-face active: distance ${distance.toFixed(2)} / ${DISTANCE_THRESHOLD} -> ${
            isMismatch ? 'mismatch' : 'match'
          }`,
        )

        if (!openRef.current) {
          // No incident yet — wait for the grace period before opening one.
          if (isMismatch) {
            belowCountRef.current += 1
            if (belowCountRef.current >= GRACE_OPEN_COUNT) {
              openRef.current = true
              belowCountRef.current = 0
              aboveCountRef.current = 0
              onMismatchRef.current(distance)
            }
          } else {
            belowCountRef.current = 0
          }
          return
        }

        // An incident is open.
        if (isMismatch) {
          // Keep the violation alive every tick (mirrors how the other
          // per-frame violations repeatedly fire while still detected).
          aboveCountRef.current = 0
          onMismatchRef.current(distance)
        } else {
          aboveCountRef.current += 1
          if (aboveCountRef.current >= RESOLVE_COUNT) {
            openRef.current = false
            aboveCountRef.current = 0
            belowCountRef.current = 0
            onMatchRef.current()
          }
        }
      } catch (error) {
        console.error('❌ Face proctoring tick error:', error)
      } finally {
        tickRunningRef.current = false
      }
    }

    ;(async () => {
      try {
        await loadModels()
        if (cancelled) return
        status('Wrong-face check: loading KYC selfie…')
        const descriptor = await fetchKycDescriptor(sessionId)
        if (cancelled) return
        if (!descriptor) {
          console.error('❌ Could not extract a face descriptor from the KYC selfie')
          status('Wrong-face check DISABLED: KYC selfie has no detectable face')
          return
        }
        kycDescriptorRef.current = descriptor
        console.log('✅ KYC baseline descriptor ready, starting face proctoring')
        status('Wrong-face check active: KYC baseline ready')
        intervalId = window.setInterval(() => {
          void tick()
        }, TICK_INTERVAL_MS)
      } catch (error) {
        console.error('❌ Face proctoring initialization failed:', error)
        status(
          `Wrong-face check FAILED to start: ${
            error instanceof Error ? error.message : String(error)
          }`,
        )
      }
    })()

    return () => {
      cancelled = true
      if (intervalId) clearInterval(intervalId)
      kycDescriptorRef.current = null
      openRef.current = false
      belowCountRef.current = 0
      aboveCountRef.current = 0
      tickRunningRef.current = false
    }
  }, [sessionId])
}
