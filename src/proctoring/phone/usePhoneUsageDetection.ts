// Orchestrator hook — ties the modular pipeline together into one detection
// loop and exposes it to React:
//
//   Detector candidates ─┐
//   MediaPipe features  ─┼─► webcam zones ─► rule engine ─► label + gates + reasons
//                        │                                       │
//                        └──────────────► overlay snapshot ◄─────┘
//                                                │
//                   temporal confirm (persistence + hysteresis)
//                                                │
//                 CONFIRMED_PHONE_USAGE ─► evidence capture + onConfirmed
//
// Only CONFIRMED_PHONE_USAGE (persisted across requiredConsecutive ticks) fires
// onConfirmed / triggers recording. Every other label is surfaced for display
// but never records a violation.

import { useEffect, useRef, useState } from 'react'
import { resolveConfig, type PhonePipelineConfig } from './config'
import { EfficientDetCandidateDetector } from './efficientDetCandidateDetector'
import { MediaPipeFeatureExtractor } from './mediapipeFeatures'
import { computeWebcamZones } from './webcamZones'
import { classifyPhoneUsage } from './ruleEngine'
import { captureEvidence } from './evidenceRecorder'
import type { OverlaySnapshot } from './overlay'
import { boxIoU } from './geometry'
import type {
  NormBox,
  PhoneEvidence,
  PhoneUsageDecision,
  PhoneUsageLabel,
} from './types'

const VIDEO_ELEMENT_ID = 'proctoring-video'

export interface UsePhoneUsageDetectionParams {
  videoElementId?: string
  enabled?: boolean
  config?: Partial<PhonePipelineConfig>
  sessionId?: string | null
  studentId?: string | null
  /** Called every tick with the latest overlay snapshot (for live drawing). */
  onFrame?: (snap: OverlaySnapshot) => void
  /** Called every tick with the raw per-frame decision. */
  onDecision?: (decision: PhoneUsageDecision) => void
  /** Called each tick while a confirmed incident is open (keeps episode alive). */
  onConfirmed?: (confidence: number, box: NormBox | null) => void
  /** Called once, on the transition into a confirmed incident, with evidence. */
  onEvidence?: (evidence: PhoneEvidence) => void
  /** Called when a confirmed incident clears (hysteresis). */
  onCleared?: () => void
  onStatus?: (message: string) => void
}

export interface UsePhoneUsageDetectionResult {
  isLoaded: boolean
  config: PhonePipelineConfig
  /** Current per-frame label. */
  label: PhoneUsageLabel
  /** True while a confirmed incident is open. */
  phoneConfirmed: boolean
  confidence: number
  error: string | null
}

export function usePhoneUsageDetection(
  params: UsePhoneUsageDetectionParams = {},
): UsePhoneUsageDetectionResult {
  const {
    videoElementId = VIDEO_ELEMENT_ID,
    enabled = true,
    sessionId = null,
    studentId = null,
  } = params

  const cfg = useRef(resolveConfig(params.config)).current

  const [isLoaded, setIsLoaded] = useState(false)
  const [label, setLabel] = useState<PhoneUsageLabel>('NO_VIOLATION')
  const [phoneConfirmed, setPhoneConfirmed] = useState(false)
  const [confidence, setConfidence] = useState(0)
  const [error, setError] = useState<string | null>(null)

  // ---- Model refs ----
  // Phone candidate detector (EfficientDet-Lite2), producing a DetectionFrame.
  const detectorRef = useRef<EfficientDetCandidateDetector | null>(null)
  const mpRef = useRef<MediaPipeFeatureExtractor | null>(null)

  // ---- Loop bookkeeping ----
  const tickRunningRef = useRef(false)
  const consecutiveRef = useRef(0)
  const missRef = useRef(0)
  const activeRef = useRef(false)
  const lastCandBoxRef = useRef<NormBox | null>(null)
  const headBoostRef = useRef(0)
  const lastTsRef = useRef(0)
  const lastDebugRef = useRef(0)

  // ---- Callback + config refs (avoid stale closures / re-subscribes) ----
  const cbRef = useRef(params)
  useEffect(() => { cbRef.current = params })
  const enabledRef = useRef(enabled)
  useEffect(() => { enabledRef.current = enabled }, [enabled])
  const idsRef = useRef({ sessionId, studentId })
  useEffect(() => { idsRef.current = { sessionId, studentId } }, [sessionId, studentId])

  const status = (m: string) => cbRef.current.onStatus?.(m)

  useEffect(() => {
    let cancelled = false
    let intervalId: number | null = null

    const getVideo = (): HTMLVideoElement | null => {
      const el = document.getElementById(videoElementId) as HTMLVideoElement | null
      if (!el) return null
      if (el.readyState < 2 || el.videoWidth === 0 || el.paused) return null
      return el
    }

    const resetIncident = () => {
      consecutiveRef.current = 0
      missRef.current = 0
      lastCandBoxRef.current = null
      headBoostRef.current = 0
      if (activeRef.current) {
        activeRef.current = false
        cbRef.current.onCleared?.()
      }
      setPhoneConfirmed(false)
      setConfidence(0)
      setLabel('NO_VIOLATION')
    }

    const tick = async () => {
      if (tickRunningRef.current) return
      tickRunningRef.current = true
      try {
        const detector = detectorRef.current
        const mp = mpRef.current
        if (!detector || !mp) return
        if (!enabledRef.current || document.hidden) { resetIncident(); return }

        const video = getVideo()
        if (!video) { resetIncident(); return }

        // Monotonic ts for MediaPipe's detectForVideo.
        let ts = performance.now()
        if (ts <= lastTsRef.current) ts = lastTsRef.current + 1
        lastTsRef.current = ts

        // ---- Run the modular pipeline ----
        const frame = await detector.detect(video)
        const rawMp = mp.detect(video, ts)
        const features = mp.extract(rawMp)
        const zones = computeWebcamZones(cfg, frame.person, features.pose)
        const decision = classifyPhoneUsage(cfg, frame, features, zones)

        const snap: OverlaySnapshot = { detections: frame, features, zones, decision }
        cbRef.current.onFrame?.(snap)
        cbRef.current.onDecision?.(decision)
        setLabel(decision.label)

        // ---- Temporal confirmation (persistence + head-down boost) ----
        const qualifies = decision.label === 'CONFIRMED_PHONE_USAGE' && decision.candidate != null
        const candBox = decision.candidate?.box ?? null

        if (features.head.headDown) {
          headBoostRef.current = Math.min(headBoostRef.current + 1, 10)
        } else {
          headBoostRef.current = 0
        }
        const headSustained = headBoostRef.current >= 3

        if (qualifies && candBox) {
          const prev = lastCandBoxRef.current
          const persists = prev === null || boxIoU(prev, candBox) >= cfg.persistIoU
          consecutiveRef.current = persists ? consecutiveRef.current + 1 : 1
          lastCandBoxRef.current = candBox
          missRef.current = 0
        } else {
          missRef.current += 1
          if (missRef.current >= cfg.clearMisses) {
            consecutiveRef.current = 0
            lastCandBoxRef.current = null
          }
        }

        // Instant detection: the geometry gate is stateless and fires on the
        // first qualifying tick. Only fall back to consecutive-tick persistence
        // when instantConfirm is disabled.
        const need = cfg.instantConfirm
          ? 1
          : headSustained
            ? Math.min(cfg.requiredConsecutive, cfg.requiredConsecutiveBoosted)
            : cfg.requiredConsecutive
        const confirmedNow = consecutiveRef.current >= need

        // Report the verifier's COMBINED confidence (hand-gate × rawScore ×
        // grip/geometry blend), not the raw detector score — that combined value
        // is what cleared the display threshold.
        const conf = decision.combinedConfidence || decision.candidate?.score || 0
        if (confirmedNow) {
          const firstConfirm = !activeRef.current
          activeRef.current = true
          setPhoneConfirmed(true)
          setConfidence(conf)
          cbRef.current.onConfirmed?.(conf, candBox)

          // Capture explainable evidence ONCE on the transition into confirmed.
          if (firstConfirm) {
            captureEvidence(video, cfg, snap, idsRef.current)
              .then((ev) => cbRef.current.onEvidence?.(ev))
              .catch((e) => console.error('❌ evidence capture failed', e))
          }
        } else if (activeRef.current && missRef.current >= cfg.clearMisses) {
          activeRef.current = false
          setPhoneConfirmed(false)
          setConfidence(0)
          cbRef.current.onCleared?.()
        }

        // ---- Status line ----
        const line =
          `Phone: ${decision.label} ` +
          `cand=${decision.candidate ? `${(conf * 100).toFixed(0)}%` : 'none'} ` +
          `gates=[${decision.passedGates.join(',') || '-'}] ` +
          `run=${consecutiveRef.current}/${need} miss=${missRef.current}` +
          `${activeRef.current ? ' CONFIRMED' : ''}`
        status(line)
        if (cfg.debug) {
          const now = Date.now()
          if (now - lastDebugRef.current >= 700) {
            lastDebugRef.current = now
            console.debug(`[phone] ${line}`, decision.reasons)
          }
        }
      } catch (err) {
        console.error('❌ Phone-usage tick error:', err)
      } finally {
        tickRunningRef.current = false
      }
    }

    ;(async () => {
      try {
        status('Phone check: loading models…')
        // EfficientDet-Lite2 (MediaPipe ObjectDetector) phone-candidate detector.
        const detector = await EfficientDetCandidateDetector.create({
          candidateScoreThreshold: cfg.candidateScoreThreshold,
          personScoreThreshold: cfg.personScoreThreshold,
          objectScoreThreshold: cfg.objectScoreThreshold,
        })
        if (cancelled) { await detector.close(); return }
        detectorRef.current = detector

        const mp = await MediaPipeFeatureExtractor.create(cfg)
        if (cancelled) { mp.close(); return }
        mpRef.current = mp

        setIsLoaded(true)
        setError(null)
        status(
          `Phone check: ready (face:${mp.hasFace ? 1 : 0} pose:${mp.hasPose ? 1 : 0} hands:${mp.hasHands ? 1 : 0})`,
        )
        intervalId = window.setInterval(() => { void tick() }, cfg.intervalMs)
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        console.error('❌ Phone-usage init failed:', err)
        setError(msg)
        status(`Phone check FAILED to start: ${msg}`)
      }
    })()

    return () => {
      cancelled = true
      if (intervalId) clearInterval(intervalId)
      void detectorRef.current?.close()
      mpRef.current?.close()
      detectorRef.current = null
      mpRef.current = null
      tickRunningRef.current = false
      consecutiveRef.current = 0
      missRef.current = 0
      activeRef.current = false
      lastCandBoxRef.current = null
      headBoostRef.current = 0
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoElementId])

  return { isLoaded, config: cfg, label, phoneConfirmed, confidence, error }
}
