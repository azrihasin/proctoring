import { useEffect, useRef } from 'react'
import * as faceapi from '@vladmandic/face-api'
import { useProctoringStore } from './proctoringStore'

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const MODEL_URL = '/models'
const KYC_URL = 'https://proctor-x-api.appricode.net/api/kyc/getPhoto'
const UPLOAD_URL = 'https://proctor-x-api.appricode.net/api/proctor/upload'
const EVENT_URL = 'https://proctor-x-api.appricode.net/api/proctor/event'
const VIDEO_ELEMENT_ID = 'proctoring-video'

const SIMILARITY_THRESHOLD = 0.7 // below => mismatch
const GRACE_OPEN_COUNT = 5 // consecutive scores below threshold required to open
const RESOLVE_COUNT = 3 // consecutive scores at/above threshold required to resolve
const TICK_INTERVAL_MS = 1000 // run a comparison every 1 second
const HEARTBEAT_INTERVAL_MS = 5 * 60 * 1000 // at most one heartbeat every 5 minutes

export interface FaceProctoringParams {
  sessionId: string | null
}

interface ActiveIncident {
  incidentId: string
  startedAt: string // ISO timestamp captured at open
  startedAtMs: number
  lastHeartbeatAt: number // epoch ms of last heartbeat (init = open time)
  scoreAtDetection: number
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

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0
  let normA = 0
  let normB = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }
  if (normA === 0 || normB === 0) return 0
  return dot / (Math.sqrt(normA) * Math.sqrt(normB))
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
// Snapshot + API senders
// ---------------------------------------------------------------------------
function captureSnapshot(video: HTMLVideoElement): Promise<Blob | null> {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      resolve(null)
      return
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.9)
  })
}

async function sendUpload(
  data: Record<string, unknown>,
  blob: Blob,
  filename: string,
): Promise<void> {
  const formData = new FormData()
  formData.append('data', new Blob([JSON.stringify(data)], { type: 'application/json' }))
  formData.append('file', blob, filename)
  const res = await fetch(UPLOAD_URL, { method: 'POST', body: formData })
  if (!res.ok) {
    throw new Error(`proctor/upload failed: ${res.status} ${res.statusText}`)
  }
}

async function sendEvent(body: Record<string, unknown>): Promise<void> {
  const res = await fetch(EVENT_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    throw new Error(`proctor/event failed: ${res.status} ${res.statusText}`)
  }
}

function makeIncidentId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `inc_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------
export function useFaceProctoring({ sessionId }: FaceProctoringParams) {
  // State-machine bookkeeping that must survive across interval ticks.
  const kycDescriptorRef = useRef<Float32Array | null>(null)
  const incidentRef = useRef<ActiveIncident | null>(null)
  const belowCountRef = useRef(0)
  const aboveCountRef = useRef(0)
  const tickRunningRef = useRef(false)

  const incidentState = useProctoringStore((s) => s.incidentState)
  const currentScore = useProctoringStore((s) => s.currentScore)
  const incidentId = useProctoringStore((s) => s.incidentId)
  const snapshotCount = useProctoringStore((s) => s.snapshotCount)

  useEffect(() => {
    if (!sessionId) {
      console.warn('⚠️ useFaceProctoring: missing sessionId, proctoring disabled')
      return
    }

    const store = useProctoringStore.getState()
    let cancelled = false
    let intervalId: number | null = null

    const getVideo = (): HTMLVideoElement | null => {
      const el = document.getElementById(VIDEO_ELEMENT_ID) as HTMLVideoElement | null
      if (!el) return null
      if (el.readyState < 2 || el.videoWidth === 0 || el.paused) return null
      return el
    }

    const openIncident = async (video: HTMLVideoElement, score: number) => {
      const startedAtMs = Date.now()
      const startedAt = new Date(startedAtMs).toISOString()
      const newIncidentId = makeIncidentId()
      incidentRef.current = {
        incidentId: newIncidentId,
        startedAt,
        startedAtMs,
        lastHeartbeatAt: startedAtMs,
        scoreAtDetection: score,
      }
      belowCountRef.current = 0
      aboveCountRef.current = 0
      store.setIncidentId(newIncidentId)
      store.setIncidentState('DETECTED')

      const blob = await captureSnapshot(video)
      if (blob) {
        store.incrementSnapshotCount()
        await sendUpload(
          {
            sessionId,
            startTime: startedAt,
            endTime: startedAt,
            incidentId: newIncidentId,
            event: 'incident_open',
            scoreAtDetection: score,
          },
          blob,
          `face-mismatch_${newIncidentId}_open.jpg`,
        )
      }
      await sendEvent({
        sessionId,
        timestamp: startedAt,
        eventType: 'face-mismatch-open',
      })
    }

    const sendHeartbeat = async (video: HTMLVideoElement, incident: ActiveIncident) => {
      const blob = await captureSnapshot(video)
      if (blob) {
        store.incrementSnapshotCount()
        await sendUpload(
          {
            sessionId,
            startTime: incident.startedAt,
            endTime: incident.startedAt,
            incidentId: incident.incidentId,
            event: 'incident_heartbeat',
            scoreAtDetection: incident.scoreAtDetection,
          },
          blob,
          `face-mismatch_${incident.incidentId}_heartbeat.jpg`,
        )
      }
      await sendEvent({
        sessionId,
        timestamp: incident.startedAt,
        eventType: 'face-mismatch-heartbeat',
      })
    }

    const resolveIncident = async (video: HTMLVideoElement, incident: ActiveIncident) => {
      const resolvedAtMs = Date.now()
      const resolvedAt = new Date(resolvedAtMs).toISOString()
      const durationMs = resolvedAtMs - incident.startedAtMs
      store.setIncidentState('RESOLVED')

      const blob = await captureSnapshot(video)
      if (blob) {
        store.incrementSnapshotCount()
        await sendUpload(
          {
            sessionId,
            startTime: incident.startedAt,
            endTime: resolvedAt,
            incidentId: incident.incidentId,
            event: 'incident_close',
            scoreAtDetection: incident.scoreAtDetection,
            resolvedAt,
            durationMs,
          },
          blob,
          `face-mismatch_${incident.incidentId}_close.jpg`,
        )
      }
      await sendEvent({
        sessionId,
        timestamp: incident.startedAt,
        eventType: 'face-mismatch-close',
      })

      incidentRef.current = null
      belowCountRef.current = 0
      aboveCountRef.current = 0
      store.setIncidentId(null)
      store.setIncidentState('CLEAR')
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

        const score = cosineSimilarity(descriptor, baseline)
        store.setCurrentScore(score)

        const incident = incidentRef.current
        if (!incident) {
          if (score < SIMILARITY_THRESHOLD) {
            belowCountRef.current += 1
            if (belowCountRef.current >= GRACE_OPEN_COUNT) {
              await openIncident(video, score)
            }
          } else {
            belowCountRef.current = 0
          }
          return
        }

        // An incident is open.
        if (score >= SIMILARITY_THRESHOLD) {
          aboveCountRef.current += 1
          if (aboveCountRef.current >= RESOLVE_COUNT) {
            await resolveIncident(video, incident)
          } else {
            store.setIncidentState('PERSISTING')
          }
        } else {
          aboveCountRef.current = 0
          store.setIncidentState('PERSISTING')
          if (Date.now() - incident.lastHeartbeatAt >= HEARTBEAT_INTERVAL_MS) {
            incident.lastHeartbeatAt = Date.now()
            await sendHeartbeat(video, incident)
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
        const descriptor = await fetchKycDescriptor(sessionId)
        if (cancelled) return
        if (!descriptor) {
          console.error('❌ Could not extract a face descriptor from the KYC selfie')
          return
        }
        kycDescriptorRef.current = descriptor
        console.log('✅ KYC baseline descriptor ready, starting face proctoring')
        intervalId = window.setInterval(() => {
          void tick()
        }, TICK_INTERVAL_MS)
      } catch (error) {
        console.error('❌ Face proctoring initialization failed:', error)
      }
    })()

    return () => {
      cancelled = true
      if (intervalId) clearInterval(intervalId)
      kycDescriptorRef.current = null
      incidentRef.current = null
      belowCountRef.current = 0
      aboveCountRef.current = 0
      tickRunningRef.current = false
      useProctoringStore.getState().reset()
    }
  }, [sessionId])

  return { incidentState, currentScore, incidentId, snapshotCount }
}
