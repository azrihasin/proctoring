import { useRef, useState, useEffect, useCallback } from 'react'
import { FaceDetector, ObjectDetector, FaceLandmarker, DrawingUtils, FilesetResolver } from '@mediapipe/tasks-vision'
import Webcam from 'react-webcam'
import { useReactMediaRecorder } from 'react-media-recorder'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { VideoPreview } from '@/components/VideoPreview'
import { fixWebmMetadata, cn } from '@/lib/utils'
import { Eye, EyeOff, Layers, Check, Download } from 'lucide-react'
import axios from 'axios'
import { useFaceProctoring } from '@/proctoring/useFaceProctoring'
// import EKYC from '@/components/EKYC'

type DetectionType = 'cell_phone' | 'multiple_faces' | 'face_not_visible' | 'tab_switch' | 'wrong_face' | 'eyes_off_screen' | null

type DetectionHistoryEntry = {
  type: DetectionType
  timestamp: Date
  score?: number
  videoBlob?: Blob
}

type ViolationEntry = {
  type: DetectionType
  violationTime: Date
  startTime: Date // violationTime - 10 seconds
  endTime: Date // violationTime + 10 seconds
  score?: number
}

type SavedVideo = {
  blob: Blob
  mime: string
  ext: 'mp4' | 'webm'
  converted: boolean
}

// Minimum gap between event-API sends / video uploads for the SAME violation
// type. A violation re-fires on every detection frame (~10x/sec) for as long as
// it is visible, so without this we would hit /event and /upload continuously.
// One event + one 10s clip per type per window is plenty for review.
const VIOLATION_COOLDOWN_MS = 30000 // 30 seconds

// Episode grace period. A violation re-fires every detection frame while it is
// visible but can also briefly drop out (a face turning, a phone moving out of
// frame for a moment). We treat a violation as one continuous "episode" until it
// has been ABSENT for longer than this gap. Only then do we close the episode
// (clearing the cooldown windows), so a 1–2s dropout does not produce a
// duplicate event/upload, while detection of the FIRST frame stays immediate.
const EPISODE_GAP_MS = 3000 // 3 seconds

// Lifecycle of a recorded item in the list:
// - 'pending'  : violation just detected, the 10s clip is still being recorded
// - 'ready'    : clip captured, blob available, can be downloaded
// - 'failed'   : recording produced no usable video (error / empty / too small)
type RecordedVideoStatus = 'pending' | 'ready' | 'failed'

type RecordedVideo = {
  id: string
  filename: string
  blob?: Blob // undefined while status === 'pending'
  mime: string
  ext: 'mp4' | 'webm'
  converted: boolean
  timestamp: Date
  type: 'exam' | 'violation'
  size: number
  mp4Blob?: Blob // Optional pre-converted MP4 blob
  status: RecordedVideoStatus
  recordingStartedAt?: number // Date.now() when recording began (drives progress bar)
  recordingDurationMs?: number // expected clip length, used to estimate progress
}

// A violation clip records for exactly 10 seconds from detection (see the stop
// timeout in startViolationRecording). Used to estimate the pending progress bar.
const VIOLATION_CLIP_DURATION_MS = 10000


export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const webcamRef = useRef<Webcam>(null)
  const recordingTimeoutRef = useRef<number | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const [modelsLoaded, setModelsLoaded] = useState(false)
  const [webcamError, setWebcamError] = useState<string | null>(null)
  const [webcamReady, setWebcamReady] = useState(false)
  const detectionCountRef = useRef<{ type: DetectionType; count: number }>({ type: null, count: 0 })
  const faceNotVisibleSinceRef = useRef<number | null>(null) // Timestamp when face first became not visible
  const multipleFacesSinceRef = useRef<number | null>(null) // Timestamp when multiple faces first appeared
  const eyesOffScreenSinceRef = useRef<number | null>(null) // Timestamp when the candidate first looked away
  const eyesOnGraceStartRef = useRef<number | null>(null) // First moment of continuous on-screen/no-face evidence; the look-away hold only resets after this persists (flicker tolerance)
  const noSingleFaceSinceRef = useRef<number | null>(null) // First moment faceCount stopped being exactly 1; baseline only drops after a sustained absence
  // Learned sign of the head-pitch axis. The transformation-matrix pitch sign
  // is camera-dependent, but eye gaze (blendshapes) is anatomically fixed, and
  // head + eyes move the SAME way during vertical glances — so accumulated
  // sign-agreement between the two tells us the camera's convention. Negative
  // evidence = pitch is inverted on this camera. Kept for the whole session.
  const pitchSignEvidenceRef = useRef<number>(0)
  // Auto-calibrated "resting posture" baseline (headYaw/pitch + gazeH/gazeV).
  // eyes_off_screen is scored as DEVIATION from this baseline, not absolute
  // angles, so a laptop camera sitting below the screen (looking at the screen
  // = looking down) doesn't read as a permanent look-away. Slowly EMA-updated
  // only while the candidate is clearly on-screen, so a real look-away never
  // pulls the baseline toward itself.
  const gazeBaselineRef = useRef<{ yaw: number; pitch: number; gazeH: number; gazeV: number } | null>(null)
  const isDetectingRef = useRef<boolean>(false) // Guard against overlapping detection cycles
  const lastTimingLogRef = useRef<number>(0) // Throttle for frame-timing diagnostics
  const faceDetectorRef = useRef<FaceDetector | null>(null)
  const objectDetectorRef = useRef<ObjectDetector | null>(null)
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null) // Head-pose + eye-gaze for eyes_off_screen
  const detectionIntervalRef = useRef<number | null>(null) // Store interval ID for detection
  
  const cellPhoneDetectionDebounceRef = useRef<number | null>(null) // For debounce
  const lastCellPhoneDetectionTimeRef = useRef<number>(0)
  const latestDetectionScoreRef = useRef<{ type: DetectionType; score: number } | null>(null) // Store latest detection score for violations
  const [detectionHistory, setDetectionHistory] = useState<DetectionHistoryEntry[]>([])
  const activeFaceNotVisibleViolationRef = useRef<number | null>(null) // Index of active face_not_visible violation in violations array
  const activeCellPhoneViolationRef = useRef<number | null>(null) // Index of active cell_phone violation in violations array
  const activeMultipleFacesViolationRef = useRef<number | null>(null) // Index of active multiple_faces violation in violations array
  const activeTabSwitchViolationRef = useRef<number | null>(null) // Index of active tab_switch violation in violations array
  const activeWrongFaceViolationRef = useRef<number | null>(null) // Index of active wrong_face violation in violations array
  const activeEyesOffScreenViolationRef = useRef<number | null>(null) // Index of active eyes_off_screen violation in violations array

  // Bottom-left warning toaster (replaces the old on-canvas detection labels).
  // One toast is pushed per NEW violation episode (never per frame — addViolation
  // only reaches the "create new violation" branch once per episode thanks to the
  // active*ViolationRef gating above), so this stays in sync with the same
  // episode boundaries the recording/log pipeline already uses.
  const [warningToasts, setWarningToasts] = useState<Array<{ id: string; type: DetectionType; label: string; count: number }>>([])
  const violationOccurrenceCountRef = useRef<Map<DetectionType, number>>(new Map())
  const MAX_VISIBLE_TOASTS = 1
  const TOAST_DURATION_MS = 4000

  const pushWarningToast = useCallback((type: DetectionType, label: string) => {
    const nextCount = (violationOccurrenceCountRef.current.get(type) || 0) + 1
    violationOccurrenceCountRef.current.set(type, nextCount)
    const id = `${type}-${Date.now()}`
    setWarningToasts(prev => {
      const next = [...prev, { id, type, label, count: nextCount }]
      return next.slice(Math.max(0, next.length - MAX_VISIBLE_TOASTS))
    })
    window.setTimeout(() => {
      setWarningToasts(prev => prev.filter(t => t.id !== id))
    }, TOAST_DURATION_MS)
  }, [])

  // Session ID from URL parameters
  const sessionIdRef = useRef<string | null>(null)
  const [sessionIdDisplay, setSessionIdDisplay] = useState<string | null>(null)

  // Parameters for the wrong-face proctoring feature (read from the URL on mount)
  const [proctorSessionId, setProctorSessionId] = useState<string | null>(null)
  // Live, human-readable status of the wrong-face checker (shown on-screen so the
  // failure point is visible without opening the browser console).
  const [faceProctoringStatus, setFaceProctoringStatus] = useState<string>('Wrong-face check: initializing…')
  
  // User ID from URL parameters
  const userIdRef = useRef<string | null>(null)
  
  // Track last log time per log message type for throttling
  const lastLogTimeRef = useRef<Map<string, number>>(new Map())
  
  // Cooldown bookkeeping — see VIOLATION_COOLDOWN_MS. These cap how often a
  // given violation type may hit the event endpoint and upload a video clip.
  const lastEventSentRef = useRef<Map<string, number>>(new Map()) // eventType -> last /event send time
  const lastVideoUploadRef = useRef<Map<DetectionType, number>>(new Map()) // type -> last clip recording start time
  // Episode bookkeeping — see EPISODE_GAP_MS. Records the last time each violation
  // type was seen, so we only close the episode (resetEventDetectionCount) after
  // the violation has been absent for longer than the grace period.
  const lastSeenRef = useRef<Map<DetectionType, number>>(new Map()) // type -> last detection time
  
  // Exam recording state
  const [isExamActive, setIsExamActive] = useState(false)
  const isExamActiveRef = useRef(false) // Ref to avoid stale closures
  const [examStartTime, setExamStartTime] = useState<Date | null>(null)
  const [recordingDuration, setRecordingDuration] = useState<number>(0) // Duration in seconds
  const examVideoChunksRef = useRef<Blob[]>([])
  const examTimerIntervalRef = useRef<number | null>(null) // For timer updates
  const examSaveIntervalRef = useRef<number | null>(null) // For periodic saving
  const lastSavedSegmentTimeRef = useRef<number>(0) // Track when last segment was saved
  const [, setViolations] = useState<ViolationEntry[]>([])
  
  // Rolling buffer for violation recording (10 seconds before detection)
  const rollingBufferRef = useRef<Array<{ blob: Blob; timestamp: number }>>([])
  const rollingBufferRecorderRef = useRef<MediaRecorder | null>(null)
  
  // Get best supported MIME type for MediaRecorder (defined before hooks)
  const getBestMimeType = (): string => {
    const mimeTypes = [
      'video/webm;codecs=vp9,opus',
      'video/webm;codecs=vp8,opus',
      'video/webm',
    ]
    
    for (const mimeType of mimeTypes) {
      if (MediaRecorder.isTypeSupported(mimeType)) {
        return mimeType
      }
    }
    
    // Fallback to generic webm (should always be supported)
    return 'video/webm'
  }
  
  // Use react-media-recorder for exam recording
  const {
    status: examRecordingStatus,
    stopRecording: stopExamRecording,
  } = useReactMediaRecorder({
    video: true,
    audio: false,
    blobPropertyBag: {
      type: "video/mp4",
    },
    mediaRecorderOptions: {
      mimeType: getBestMimeType(),
      videoBitsPerSecond: 1500000,
    },
  })

  // Use react-media-recorder for violation recording
  const {
    status: violationRecordingStatus,
    stopRecording: stopViolationRecordingHook,
  } = useReactMediaRecorder({
    video: true,
    audio: false,
    blobPropertyBag: {
      type: "video/mp4",
    },
    mediaRecorderOptions: {
      mimeType: getBestMimeType(),
      videoBitsPerSecond: 1500000,
    },
  })
  
  // Track active recordings by violation type to prevent duplicate recordings
  const activeRecordingByTypeRef = useRef<Map<DetectionType, boolean>>(new Map())
  
  // Separate MediaRecorders for each violation type (can record same stream simultaneously)
  const violationRecordersRef = useRef<Map<DetectionType, MediaRecorder>>(new Map())
  const violationChunksByTypeRef = useRef<Map<DetectionType, Blob[]>>(new Map())
  const violationTimeoutsRef = useRef<Map<DetectionType, number>>(new Map())
  const violationDetectionTimesRef = useRef<Map<DetectionType, Date>>(new Map())
  // Id of the "pending" list entry created when a violation is first detected,
  // so onstop can finalize (or fail) the very same item it created.
  const violationPendingIdsRef = useRef<Map<DetectionType, string>>(new Map())
  
  // UI state
  // Per-detection overlay toggles. Each controls whether that detector's boxes /
  // HUD are DRAWN on the webcam canvas (detection/violation logic is unaffected).
  // The detection loop runs from a stale interval closure, so it reads these from
  // refs — the state copies below only drive the checkbox menu UI.
  const [showPersonOverlay, setShowPersonOverlay] = useState(false)
  const [showPhoneOverlay, setShowPhoneOverlay] = useState(false)
  const [showEyesOverlay, setShowEyesOverlay] = useState(false)
  const showPersonOverlayRef = useRef(false)
  const showPhoneOverlayRef = useRef(false)
  const showEyesOverlayRef = useRef(false)
  const [isOverlayMenuOpen, setIsOverlayMenuOpen] = useState(false)
  const overlayMenuRef = useRef<HTMLDivElement>(null)

  // Mirror the overlay toggle state into refs so the (stale-closure) detection
  // loop always draws according to the latest choice.
  useEffect(() => { showPersonOverlayRef.current = showPersonOverlay }, [showPersonOverlay])
  useEffect(() => { showPhoneOverlayRef.current = showPhoneOverlay }, [showPhoneOverlay])
  useEffect(() => { showEyesOverlayRef.current = showEyesOverlay }, [showEyesOverlay])

  // Close the overlay checkbox menu when clicking anywhere outside it.
  useEffect(() => {
    if (!isOverlayMenuOpen) return
    const handleClickOutside = (event: MouseEvent) => {
      if (overlayMenuRef.current && !overlayMenuRef.current.contains(event.target as Node)) {
        setIsOverlayMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isOverlayMenuOpen])
  const [logEntries, setLogEntries] = useState<string[]>([])
  const [isViewVisible, setIsViewVisible] = useState(false) // Toggle visibility for camera, log, and recorded list
  
  // Recorded videos list
  const [recordedVideos, setRecordedVideos] = useState<RecordedVideo[]>([])

  // Ticking clock used to animate the "pending" progress bars. Only runs while
  // at least one item is still recording, so it costs nothing when idle.
  const [nowTick, setNowTick] = useState(() => Date.now())
  const hasPendingVideos = recordedVideos.some(v => v.status === 'pending')
  useEffect(() => {
    if (!hasPendingVideos) return
    const intervalId = window.setInterval(() => setNowTick(Date.now()), 200)
    return () => clearInterval(intervalId)
  }, [hasPendingVideos])

  // Refs for auto-scrolling
  const logSectionRef = useRef<HTMLTextAreaElement>(null)
  const recordingListRef = useRef<HTMLDivElement>(null)

  // Log when recorded videos changes
  useEffect(() => {
    // Removed console.log statements
  }, [recordedVideos])

  // Auto-scroll log section to bottom when new entries are added
  useEffect(() => {
    if (logSectionRef.current) {
      logSectionRef.current.scrollTop = logSectionRef.current.scrollHeight
    }
  }, [logEntries])

  // Auto-scroll recording list to bottom when new videos are added
  useEffect(() => {
    if (recordingListRef.current) {
      recordingListRef.current.scrollTop = recordingListRef.current.scrollHeight
    }
  }, [recordedVideos])

  // Auto-start flag — ensures exam starts only once when both webcam and models are ready
  const hasAutoStartedRef = useRef(false)

  // eKYC state
  // const [eKYCCompleted, setEKYCCompleted] = useState(false)

  // Extract URL parameters and create session ID
  const extractSessionId = useCallback(() => {
    const urlParams = new URLSearchParams(window.location.search)
    const idNo = urlParams.get('idNo')
    const sessionIdParam = urlParams.get('sessionId') // Prefer explicit sessionId param
    const userId = urlParams.get('userId') || idNo // Use userId param if available, otherwise use idNo
    const sessionId = sessionIdParam || idNo // Fall back to idNo when sessionId is absent

    if (sessionId) {
      sessionIdRef.current = sessionId
      setSessionIdDisplay(sessionId)
      userIdRef.current = userId || sessionId // Use userId if provided, otherwise use sessionId
      return sessionId
    } else {
      console.warn('⚠️ Missing URL parameter: neither sessionId nor idNo found')
      // Set session ID to null when parameter is missing
      sessionIdRef.current = null
      setSessionIdDisplay(null)
      userIdRef.current = userId || null
      return null
    }
  }, [])

  // Get video element from webcam ref
  const getVideoElement = useCallback(() => {
    return webcamRef.current?.video
  }, [])

  // Get media stream reliably - tries multiple sources
  const getMediaStream = useCallback((): MediaStream | null => {
    // First try mediaStreamRef
    if (mediaStreamRef.current) {
      if (mediaStreamRef.current.active) {
        return mediaStreamRef.current
      } else {
        console.warn('⚠️ mediaStreamRef stream is not active')
      }
    } else {
      console.warn('⚠️ mediaStreamRef.current is null')
    }
    
    // Then try video element
    const videoElement = webcamRef.current?.video
    
    if (videoElement && videoElement.srcObject) {
      const stream = videoElement.srcObject as MediaStream
      if (stream && stream.active) {
        // Update mediaStreamRef for future use
        mediaStreamRef.current = stream
        return stream
      } else {
        console.warn('⚠️ Stream from video element is not active')
      }
    }
    
    console.error('❌ No active stream found from any source')
    return null
  }, [])

  // Save video in original format (no conversion)
  const saveVideo = useCallback(async (webmBlob: Blob): Promise<SavedVideo> => {
    // Fix WebM metadata to preserve duration
    const fixedWebmBlob = await fixWebmMetadata(webmBlob)
    const mimeType = fixedWebmBlob.type || getBestMimeType()
    const ext = mimeType.includes('webm') ? 'webm' : 'mp4'
    
    return { blob: fixedWebmBlob, mime: mimeType, ext: ext as 'mp4' | 'webm', converted: false }
  }, [getBestMimeType])


  // Handle webcam user media callback
  const handleUserMedia = useCallback((stream: MediaStream) => {
    mediaStreamRef.current = stream
    setWebcamReady(true)
    setWebcamError(null)
  }, [])

  // Handle webcam error
  const handleUserMediaError = useCallback((error: string | DOMException) => {
    console.error('Error initializing webcam:', error)
    const errorMessage = typeof error === 'string' ? error : error.message
    
    if (errorMessage.includes('NotAllowedError') || errorMessage.includes('Permission denied')) {
      setWebcamError('Camera access denied. Please allow camera access in your browser settings.')
    } else if (errorMessage.includes('NotFoundError') || errorMessage.includes('No camera')) {
      setWebcamError('No camera found. Please connect a camera and try again.')
    } else {
      setWebcamError('Failed to access camera. Please check your camera permissions.')
    }
    setWebcamReady(false)
  }, [])



  // API function to send log event
  const sendLogToAPI = useCallback(async (eventType: string, timestamp: Date) => {
    if (!sessionIdRef.current) {
      console.warn('⚠️ No session ID available, skipping API call')
      return
    }

    try {
      // Format timestamp as ISO string without milliseconds (e.g., "2018-12-30T19:34:50")
      const timestampStr = timestamp.toISOString().slice(0, 19)

      const body = {
        sessionId: sessionIdRef.current, // UUID string — do not coerce to number
        timestamp: timestampStr,
        eventType: eventType
      }

      // Console log for log event being sent to API
      console.log('📝 Log event sent to API:', body)

      // Send to real API endpoint
      const response = await fetch('https://proctor-x-api.appricode.net/api/proctor/event', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body)
      })
      
      if (!response.ok) {
        throw new Error(`API call failed: ${response.statusText}`)
      }
      
      const result = await response.json()
      console.log('✅ Log event sent successfully:', result)
    } catch (error) {
      console.error('❌ Error sending log to API:', error)
      // Don't add log entry for API errors to avoid infinite loops
    }
  }, [])

  // Map violation types to eventType format for API
  const getEventTypeFromMessage = useCallback((message: string): string | null => {
    const messageLower = message.toLowerCase()
    if (messageLower.includes('cell phone') || messageLower.includes('phone')) {
      return 'phone'
    } else if (messageLower.includes('face not visible') || messageLower.includes('face-not-visible')) {
      return 'face-not-visible'
    } else if (messageLower.includes('multiple faces') || messageLower.includes('multiple-faces')) {
      return 'multiple-faces'
    } else if (messageLower.includes('tab switch') || messageLower.includes('tab-switch')) {
      return 'tab-switch'
    } else if (messageLower.includes('face mismatch') || messageLower.includes('wrong face')) {
      return 'face-mismatch'
    } else if (messageLower.includes('eyes off screen') || messageLower.includes('eyes-off-screen')) {
      return 'eyes-off-screen'
    }
    return null
  }, [])

  // Map DetectionType to eventType format
  const getEventTypeFromDetectionType = useCallback((detectionType: DetectionType): string | null => {
    if (detectionType === 'cell_phone') {
      return 'phone'
    } else if (detectionType === 'face_not_visible') {
      return 'face-not-visible'
    } else if (detectionType === 'multiple_faces') {
      return 'multiple-faces'
    } else if (detectionType === 'tab_switch') {
      return 'tab-switch'
    } else if (detectionType === 'wrong_face') {
      return 'face-mismatch'
    } else if (detectionType === 'eyes_off_screen') {
      return 'eyes-off-screen'
    }
    return null
  }, [])

  // Clear the cooldown windows when a violation ends, so that a genuinely NEW
  // occurrence of the same type later is treated fresh (event + clip sent again)
  // instead of being suppressed by a stale cooldown timestamp.
  const resetEventDetectionCount = useCallback((detectionType: DetectionType) => {
    const eventType = getEventTypeFromDetectionType(detectionType)
    if (eventType) {
      lastEventSentRef.current.delete(eventType)
      lastLogTimeRef.current.delete(eventType) // also clear the visible-log throttle so the log line tallies with the new clip
    }
    lastVideoUploadRef.current.delete(detectionType)
  }, [getEventTypeFromDetectionType])

  // Add log entry helper with throttling for repeating logs
  const addLogEntry = useCallback((message: string) => {
    // Only add logs when exam is active
    if (!isExamActiveRef.current) {
      return
    }

    const eventType = getEventTypeFromMessage(message)

    // Throttle the visible log by a STABLE key. Some messages embed changing
    // text (e.g. the cell-phone confidence "%"), which would otherwise produce a
    // different key every frame and defeat throttling entirely — so for known
    // violation types we key by the eventType, not the raw message.
    const throttleKey = eventType || message

    const now = Date.now()
    const lastLogTime = lastLogTimeRef.current.get(throttleKey) || 0
    // Match the recording/clip cooldown so a continuous violation produces one log
    // line per recorded clip — keeping the log list tallied with the video list.
    const LOG_THROTTLE_INTERVAL_MS = VIOLATION_COOLDOWN_MS

    // For repeating logs, only show if the cooldown window has passed since last occurrence
    if (lastLogTime > 0 && now - lastLogTime < LOG_THROTTLE_INTERVAL_MS) {
      return
    }
    lastLogTimeRef.current.set(throttleKey, now)

    const timestamp = new Date().toLocaleString('en-US', {
      month: '2-digit',
      day: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    })
    const logMessage = `${timestamp} ${message}`
    setLogEntries(prev => [...prev.slice(-99), logMessage]) // Keep last 100 entries

    // Send the violation event to the API at most once per cooldown window per
    // event type. (The matching video clip is uploaded on the same cooldown,
    // gated separately inside startViolationRecording.)
    if (eventType) {
      const lastEventSent = lastEventSentRef.current.get(eventType) || 0
      if (now - lastEventSent >= VIOLATION_COOLDOWN_MS) {
        lastEventSentRef.current.set(eventType, now)
        sendLogToAPI(eventType, new Date()).catch((error) => {
          console.error('Error sending log to API:', error)
        })
      }
    }
  }, [getEventTypeFromMessage, sendLogToAPI])

  // API function to send video
  const sendVideoToAPI = useCallback(async (
    videoBlob: Blob,
    eventType: string,
    startTime: Date,
    endTime: Date
  ) => {
    if (!sessionIdRef.current) {
      console.warn('⚠️ No session ID available, skipping API call')
      return
    }

    try {
      const url = "https://proctor-x-api.appricode.net/api/proctor/upload"

      // Format timestamps as ISO string without milliseconds (e.g., "2025-12-30T19:34:50")
      const startTimeStr = startTime.toISOString().slice(0, 19)
      const endTimeStr = endTime.toISOString().slice(0, 19)

      const payload = {
        sessionId: sessionIdRef.current,
        startTime: startTimeStr,
        endTime: endTimeStr,
      }

      const formData = new FormData()

      // Part "data" as application/json (like curl: -F 'data=...;type=application/json')
      formData.append(
        "data",
        new Blob([JSON.stringify(payload)], { type: "application/json" })
      )

      // Part "file"
      const fileExtension = videoBlob.type.includes('webm') ? 'webm' : 'mp4'
      const fileName = `${eventType}_${startTimeStr.replace(/[:.]/g, '-')}.${fileExtension}`
      formData.append("file", videoBlob, fileName)

      // Console log for recorded video being sent to API
      console.log('📹 Recorded video sent to API:', {
        blobSize: videoBlob.size,
        blobSizeMB: (videoBlob.size / (1024 * 1024)).toFixed(2) + ' MB',
        mimeType: videoBlob.type,
        eventType: eventType,
        sessionId: sessionIdRef.current,
        startTime: startTimeStr,
        endTime: endTimeStr,
        filename: fileName
      })

      const res = await axios.post(url, formData, {
        headers: {
          Accept: "application/json",
          // DO NOT set Content-Type here; axios will set correct multipart boundary
        },
      })

      console.log('✅ Video uploaded successfully:', res.data)
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const errorMessage = error.response 
          ? `Upload failed ${error.response.status}: ${error.response.data || error.message}`
          : error.message
        console.error('❌ Error sending video to API:', errorMessage)
      } else {
        console.error('❌ Error sending video to API:', error)
      }
      // Error is logged to console but not added to log section
    }
  }, [])

  // Add recorded video to list. Returns the generated id so callers can update
  // the entry later (e.g. swap in the transcoded MP4 once it's ready).
  const addRecordedVideo = useCallback((savedVideo: SavedVideo, filename: string, type: 'exam' | 'violation'): string => {
    // Store the blob directly - React state will keep it in memory
    // The blob should remain valid as long as it's in state
    const id = `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    const recordedVideo: RecordedVideo = {
      id,
      filename,
      blob: savedVideo.blob, // Store blob directly
      mime: savedVideo.mime,
      ext: savedVideo.ext,
      converted: savedVideo.converted,
      timestamp: new Date(),
      type,
      size: savedVideo.blob.size,
      mp4Blob: savedVideo.converted && savedVideo.ext === 'mp4' ? savedVideo.blob : undefined,
      status: 'ready',
    }

    setRecordedVideos(prev => [...prev, recordedVideo])
    return id
  }, [])

  // Add a placeholder entry the moment a violation is detected, so the item
  // shows up in the list right away with a "pending" status + progress bar.
  // The matching clip is filled in later via finalizeRecordedVideo / markRecordedVideoFailed.
  const addPendingRecordedVideo = useCallback((filename: string, ext: 'mp4' | 'webm'): string => {
    const id = `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    const recordedVideo: RecordedVideo = {
      id,
      filename,
      blob: undefined,
      mime: '',
      ext,
      converted: false,
      timestamp: new Date(),
      type: 'violation',
      size: 0,
      status: 'pending',
      recordingStartedAt: Date.now(),
      recordingDurationMs: VIOLATION_CLIP_DURATION_MS,
    }

    setRecordedVideos(prev => [...prev, recordedVideo])
    return id
  }, [])

  // Swap a pending entry's placeholder for the finished clip and mark it ready.
  const finalizeRecordedVideo = useCallback((id: string, savedVideo: SavedVideo, filename: string) => {
    setRecordedVideos(prev =>
      prev.map(v =>
        v.id === id
          ? {
              ...v,
              filename,
              blob: savedVideo.blob,
              mime: savedVideo.mime,
              ext: savedVideo.ext,
              converted: savedVideo.converted,
              size: savedVideo.blob.size,
              mp4Blob: savedVideo.converted && savedVideo.ext === 'mp4' ? savedVideo.blob : undefined,
              status: 'ready',
            }
          : v
      )
    )
  }, [])

  // Mark a pending entry as failed (no usable clip was produced).
  const markRecordedVideoFailed = useCallback((id: string) => {
    setRecordedVideos(prev =>
      prev.map(v => (v.id === id ? { ...v, status: 'failed' } : v))
    )
  }, [])

  // MediaRecorder for chunk access (needed for periodic saving)
  const examChunkRecorderRef = useRef<MediaRecorder | null>(null)

  // Start exam recording
  const handleStartExam = useCallback(() => {
    const videoElement = getVideoElement()
    if (!videoElement || !videoElement.srcObject) {
      console.error('Video stream not available for exam recording')
      return
    }

    const stream = videoElement.srcObject as MediaStream
    if (!stream) {
      console.error('Media stream not initialized')
      return
    }

    try {
      // Ensure mediaStreamRef is set to the current stream
      mediaStreamRef.current = stream
      
      // Verify stream is active
      if (!stream.active) {
        console.warn('⚠️ Stream is not active!')
      }
      
      setIsExamActive(true)
      isExamActiveRef.current = true // Update ref immediately
      const startTime = new Date()
      setExamStartTime(startTime)
      setRecordingDuration(0)
      setViolations([])
      setLogEntries([]) // Clear all logs when exam starts
      lastLogTimeRef.current.clear() // Clear log throttle map when exam starts
      activeFaceNotVisibleViolationRef.current = null // Reset active face_not_visible violation
      activeCellPhoneViolationRef.current = null // Reset active cell_phone violation
      activeMultipleFacesViolationRef.current = null // Reset active multiple_faces violation
      activeTabSwitchViolationRef.current = null // Reset active tab_switch violation
      activeWrongFaceViolationRef.current = null // Reset active wrong_face violation
      activeEyesOffScreenViolationRef.current = null // Reset active eyes_off_screen violation
      eyesOffScreenSinceRef.current = null // Reset the eyes_off_screen gating timer
      eyesOnGraceStartRef.current = null
      noSingleFaceSinceRef.current = null
      gazeBaselineRef.current = null // Recalibrate resting posture at exam start
      examVideoChunksRef.current = []
      lastSavedSegmentTimeRef.current = Date.now()

      // ⚠️ IMPORTANT: NO RECORDING STARTS HERE - ONLY WHEN VIOLATIONS ARE DETECTED
      // startExamRecording() // DISABLED - NO EXAM RECORDING
      // startRollingBuffer(stream) // DISABLED - NO PRE-RECORDING BUFFER
      // NO MediaRecorder is created here
      // NO recording is started here
      // Recording ONLY starts in startViolationRecording() when violations are detected

      // Start timer interval (update every second) - just for display
      examTimerIntervalRef.current = window.setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime.getTime()) / 1000)
        setRecordingDuration(elapsed)
      }, 1000)

      // Reset per-type cooldown windows when exam starts
      lastEventSentRef.current.clear()
      lastVideoUploadRef.current.clear()
      lastSeenRef.current.clear()

      // NO exam video segments - NO periodic recording
    } catch (error) {
      console.error('Error starting exam recording:', error)
      setIsExamActive(false)
    }
  }, [addLogEntry])

  // Validate blob duration and metadata
  const validateBlobDuration = useCallback((blob: Blob): Promise<{ duration: number; isValid: boolean }> => {
    return new Promise((resolve) => {
      const url = URL.createObjectURL(blob)
      const v = document.createElement('video')
      v.src = url
      v.muted = true

      v.onloadedmetadata = () => {
        const duration = v.duration
        const isValid = isFinite(duration) && duration > 0 && !isNaN(duration)
        URL.revokeObjectURL(url)
        resolve({ duration, isValid })
      }

      v.onerror = (e) => {
        console.error('❌ Video blob validation error:', e)
        URL.revokeObjectURL(url)
        resolve({ duration: NaN, isValid: false })
      }

      // Timeout after 5 seconds
      setTimeout(() => {
        if (v.readyState === 0) {
          console.warn('⏱️ Blob validation timeout')
          URL.revokeObjectURL(url)
          resolve({ duration: NaN, isValid: false })
        }
      }, 5000)
    })
  }, [])

  // Download the recorded clip as-is (WebM). We no longer transcode to MP4 —
  // the clip is downloaded exactly as it was recorded and uploaded.
  const downloadVideo = useCallback((video: RecordedVideo) => {
    try {
      if (!video.blob || video.blob.size === 0) {
        console.error('Video blob is null or empty')
        return
      }

      const url = URL.createObjectURL(video.blob)
      const a = document.createElement('a')
      a.href = url
      a.download = video.filename
      a.style.display = 'none'
      document.body.appendChild(a)

      // Trigger download
      a.click()

      // Clean up after a short delay
      setTimeout(() => {
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
      }, 100)
    } catch (error) {
      console.error('Error downloading video:', error)
    }
  }, [])

  // End exam and save final video
  const handleEndExam = useCallback(async () => {
    if (!isExamActive) {
      return
    }

    try {
      // Stop timer interval
      if (examTimerIntervalRef.current) {
        clearInterval(examTimerIntervalRef.current)
        examTimerIntervalRef.current = null
      }

      // Stop save interval
      if (examSaveIntervalRef.current) {
        clearInterval(examSaveIntervalRef.current)
        examSaveIntervalRef.current = null
      }

      // Stop rolling buffer recorder
      if (rollingBufferRecorderRef.current && rollingBufferRecorderRef.current.state !== 'inactive') {
        rollingBufferRecorderRef.current.stop()
        rollingBufferRecorderRef.current = null
      }
      
      // Stop all active violation recorders
      for (const [, recorder] of violationRecordersRef.current.entries()) {
        if (recorder && recorder.state === 'recording') {
          recorder.stop()
        }
      }
      
      // Clear all timeouts
      for (const timeoutId of violationTimeoutsRef.current.values()) {
        if (timeoutId) {
          clearTimeout(timeoutId)
        }
      }
      
      // Wait a bit for violation videos to be processed
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // Clear all violation recording refs
      violationRecordersRef.current.clear()
      violationChunksByTypeRef.current.clear()
      violationDetectionTimesRef.current.clear()
      violationTimeoutsRef.current.clear()
      activeRecordingByTypeRef.current.clear()
      
      // Stop violation recorder if active (using react-media-recorder) - keeping for compatibility
      if (violationRecordingStatus === 'recording') {
        stopViolationRecordingHook()
        await new Promise(resolve => setTimeout(resolve, 1000))
      }
      
      // Don't stop exam recording - we're not recording the exam anymore
      // stopExamRecording() // Disabled - only recording violations
      
      // Reset per-type cooldown windows when exam ends
      lastEventSentRef.current.clear()
      lastVideoUploadRef.current.clear()
      lastSeenRef.current.clear()

      // No final exam video to save - only violation recordings are saved

        setIsExamActive(false)
        isExamActiveRef.current = false // Update ref
        setExamStartTime(null)
        setRecordingDuration(0)
        examVideoChunksRef.current = []
        examChunkRecorderRef.current = null
        rollingBufferRef.current = []
    } catch (error) {
      console.error('Error ending exam:', error)
      setIsExamActive(false)
      setRecordingDuration(0)
    }
  }, [isExamActive, recordingDuration, saveVideo, addLogEntry, addRecordedVideo, getBestMimeType, stopExamRecording, stopViolationRecordingHook, violationRecordingStatus])

  // Start violation recording (automatically when violation detected) - DIRECT APPROACH
  const startViolationRecording = useCallback(async (stream: MediaStream, detectionTime: Date, violationType: DetectionType) => {
    if (!violationType) return
    
    // Check if a recording for this violation type is already in progress
    if (activeRecordingByTypeRef.current.get(violationType) === true) {
      return
    }

    // Cooldown: a violation fires on every detection frame while it persists, so
    // don't start (and later upload) a fresh clip if we already recorded one for
    // this type within the cooldown window. Caps uploads at one clip per window.
    const lastUpload = lastVideoUploadRef.current.get(violationType) || 0
    if (Date.now() - lastUpload < VIOLATION_COOLDOWN_MS) {
      return
    }

    // Validate stream
    if (!stream || !stream.active) {
      console.error('❌ Stream is not active or invalid')
      return
    }
    
    const videoTracks = stream.getVideoTracks()
    if (videoTracks.length === 0) {
      console.error('❌ No video tracks in stream')
      return
    }
    
    // Check if video track is enabled
    const videoTrack = videoTracks[0]
    if (videoTrack.readyState !== 'live') {
      console.error(`❌ Video track not live: ${videoTrack.readyState}`)
      return
    }
    
    // Mark this violation type as being recorded and open the cooldown window
    // now (at recording start) so concurrent frames don't queue more clips.
    activeRecordingByTypeRef.current.set(violationType, true)
    lastVideoUploadRef.current.set(violationType, Date.now())

    // Initialize chunks array for this violation type
    violationChunksByTypeRef.current.set(violationType, [])
    violationDetectionTimesRef.current.set(violationType, detectionTime)

    // Add a "pending" item to the list immediately so the violation shows up
    // right away (with a progress bar) while the 10s clip is still recording.
    const pendingExt: 'mp4' | 'webm' = getBestMimeType().includes('webm') ? 'webm' : 'mp4'
    const pendingTimestamp = detectionTime.toISOString().replace(/[:.]/g, '-').slice(0, -5)
    const pendingViolationTypeStr = violationType === 'cell_phone' ? 'cell_phone_detection' :
                                    violationType === 'multiple_faces' ? 'multiple_faces_detection' :
                                    violationType === 'face_not_visible' ? 'face_not_visible_detection' :
                                    violationType === 'tab_switch' ? 'tab_switch_detection' :
                                    violationType === 'wrong_face' ? 'wrong_face_detection' :
                                    violationType === 'eyes_off_screen' ? 'eyes_off_screen_detection' :
                                    'violation'
    const pendingFilename = `${pendingViolationTypeStr}_${pendingTimestamp}.${pendingExt}`
    const pendingId = addPendingRecordedVideo(pendingFilename, pendingExt)
    violationPendingIdsRef.current.set(violationType, pendingId)

    try {
      // Clear any existing timeout for this violation type
      const existingTimeout = violationTimeoutsRef.current.get(violationType)
      if (existingTimeout) {
        clearTimeout(existingTimeout)
        violationTimeoutsRef.current.delete(violationType)
      }
      
      // Create MediaRecorder for this specific violation type
      const mimeType = getBestMimeType()
      const options = { mimeType, videoBitsPerSecond: 1500000 }
      
      // Check if MediaRecorder is supported
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        console.warn(`⚠️ MIME type ${mimeType} not supported, using default`)
      }
      
      const recorder = new MediaRecorder(stream, options)
      violationRecordersRef.current.set(violationType, recorder)
      
      // Capture violationType in closure for event handlers
      const currentViolationType = violationType
      const chunks = violationChunksByTypeRef.current.get(violationType) || []
      violationChunksByTypeRef.current.set(violationType, chunks)
      
      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          const currentChunks = violationChunksByTypeRef.current.get(currentViolationType) || []
          currentChunks.push(event.data)
          violationChunksByTypeRef.current.set(currentViolationType, currentChunks)
        } else {
          console.warn(`⚠️ Empty chunk received for ${currentViolationType}`)
        }
      }
      
      recorder.onstop = async () => {
        // Wait a moment to ensure all data is available
        await new Promise(resolve => setTimeout(resolve, 100))

        // Flip this type's pending list entry to "failed" (or drop it if absent).
        const failPendingForType = (type: DetectionType) => {
          const id = violationPendingIdsRef.current.get(type)
          if (id) {
            markRecordedVideoFailed(id)
            violationPendingIdsRef.current.delete(type)
          }
        }

        const chunks = violationChunksByTypeRef.current.get(currentViolationType) || []
        const detectionTime = violationDetectionTimesRef.current.get(currentViolationType)
        
        // Process video directly here to avoid closure issues
        if (chunks.length === 0) {
          console.error(`❌❌❌ NO CHUNKS TO PROCESS for ${currentViolationType} - RECORDING FAILED`)
          activeRecordingByTypeRef.current.set(currentViolationType, false)
          violationRecordersRef.current.delete(currentViolationType)
          violationChunksByTypeRef.current.delete(currentViolationType)
          violationDetectionTimesRef.current.delete(currentViolationType)
          failPendingForType(currentViolationType)
          return
        }

        if (!detectionTime) {
          console.error(`❌ Missing detection time for ${currentViolationType}`)
          activeRecordingByTypeRef.current.set(currentViolationType, false)
          violationRecordersRef.current.delete(currentViolationType)
          violationChunksByTypeRef.current.delete(currentViolationType)
          violationDetectionTimesRef.current.delete(currentViolationType)
          failPendingForType(currentViolationType)
          return
        }
        
        try {
          const currentMimeType = getBestMimeType()
          
          const blob = new Blob(chunks, { type: currentMimeType })
          
          if (blob.size < 50_000) {
            console.error(`❌ Video too small for ${currentViolationType}: ${blob.size} bytes (expected at least 50KB)`)
            activeRecordingByTypeRef.current.set(currentViolationType, false)
            violationRecordersRef.current.delete(currentViolationType)
            violationChunksByTypeRef.current.delete(currentViolationType)
            violationDetectionTimesRef.current.delete(currentViolationType)
            failPendingForType(currentViolationType)
            return
          }
          
          // Fix WebM metadata (duration) so the clip is seekable/playable
          const fixedBlob = await fixWebmMetadata(blob)

          const baseExt: 'mp4' | 'webm' = currentMimeType.includes('webm') ? 'webm' : 'mp4'
          const timestamp = detectionTime.toISOString().replace(/[:.]/g, '-').slice(0, -5)
          const violationTypeStr = currentViolationType === 'cell_phone' ? 'cell_phone_detection' :
                                   currentViolationType === 'multiple_faces' ? 'multiple_faces_detection' :
                                   currentViolationType === 'face_not_visible' ? 'face_not_visible_detection' :
                                   currentViolationType === 'tab_switch' ? 'tab_switch_detection' :
                                   currentViolationType === 'wrong_face' ? 'wrong_face_detection' :
                                   currentViolationType === 'eyes_off_screen' ? 'eyes_off_screen_detection' :
                                   'violation'
          const baseFilename = `${violationTypeStr}_${timestamp}`

          // Fill in the pending list entry created at recording start (swap the
          // placeholder for the finished clip and flip it to "ready"). If for any
          // reason the pending id is gone, fall back to adding a fresh entry.
          const pendingId = violationPendingIdsRef.current.get(currentViolationType)
          if (pendingId) {
            finalizeRecordedVideo(
              pendingId,
              { blob: fixedBlob, mime: currentMimeType, ext: baseExt, converted: false },
              `${baseFilename}.${baseExt}`
            )
            violationPendingIdsRef.current.delete(currentViolationType)
          } else {
            addRecordedVideo(
              { blob: fixedBlob, mime: currentMimeType, ext: baseExt, converted: false },
              `${baseFilename}.${baseExt}`,
              'violation'
            )
          }

          // Map violation type to eventType format
          const eventTypeMap: Record<string, string> = {
            'cell_phone': 'phone',
            'face_not_visible': 'face-not-visible',
            'multiple_faces': 'multiple-faces',
            'tab_switch': 'tab-switch',
            'wrong_face': 'face-mismatch',
            'eyes_off_screen': 'eyes-off-screen'
          }
          const eventType = eventTypeMap[currentViolationType] || currentViolationType || 'unknown'

          // Recording starts at detectionTime and stops after 10 seconds
          const startTime = detectionTime
          const endTime = new Date(detectionTime.getTime() + 10000)

          // Clear refs and recording flag for this violation type now that the
          // clip is captured — background transcode/upload below is independent.
          activeRecordingByTypeRef.current.set(currentViolationType, false)
          violationRecordersRef.current.delete(currentViolationType)
          violationChunksByTypeRef.current.delete(currentViolationType)
          violationDetectionTimesRef.current.delete(currentViolationType)
          violationTimeoutsRef.current.delete(currentViolationType)

          // Upload the recorded clip as WebM (no MP4 transcode). The clip is
          // stored and uploaded exactly as MediaRecorder produced it.
          void sendVideoToAPI(fixedBlob, eventType, startTime, endTime).catch((error) => {
            console.error('Error sending video to API:', error)
          })
        } catch (error) {
          console.error(`❌❌❌ ERROR in onstop processing for ${currentViolationType}:`, error)
          // Clear recording flag even on error
          activeRecordingByTypeRef.current.set(currentViolationType, false)
          violationRecordersRef.current.delete(currentViolationType)
          violationChunksByTypeRef.current.delete(currentViolationType)
          violationDetectionTimesRef.current.delete(currentViolationType)
          violationTimeoutsRef.current.delete(currentViolationType)
          failPendingForType(currentViolationType)
        }
      }
      
      recorder.onerror = (event) => {
        console.error(`❌ Violation recorder error for ${violationType}:`, event)
        activeRecordingByTypeRef.current.set(violationType, false)
        violationRecordersRef.current.delete(violationType)
        violationChunksByTypeRef.current.delete(violationType)
        violationDetectionTimesRef.current.delete(violationType)
        violationTimeoutsRef.current.delete(violationType)
        const pid = violationPendingIdsRef.current.get(violationType)
        if (pid) {
          markRecordedVideoFailed(pid)
          violationPendingIdsRef.current.delete(violationType)
        }
      }
      
      // Start recording
      try {
        recorder.start(1000) // 1 second chunks
      } catch (startError) {
        console.error(`❌ Error starting recorder for ${violationType}:`, startError)
        activeRecordingByTypeRef.current.set(violationType, false)
        violationRecordersRef.current.delete(violationType)
        violationChunksByTypeRef.current.delete(violationType)
        violationDetectionTimesRef.current.delete(violationType)
        const pid = violationPendingIdsRef.current.get(violationType)
        if (pid) {
          markRecordedVideoFailed(pid)
          violationPendingIdsRef.current.delete(violationType)
        }
        throw startError
      }
      
      // Schedule stop after exactly 10 seconds from detection time
      // This ensures recording continues for 10 seconds including during the violation
      const timeoutId = window.setTimeout(async () => {
        const recorder = violationRecordersRef.current.get(violationType)
        if (recorder && recorder.state === 'recording') {
          // Request any pending data before stopping
          recorder.requestData()
          // Wait a bit for data to be available
          await new Promise(resolve => setTimeout(resolve, 200))
          recorder.stop()
        } else {
          console.warn(`⚠️ ${violationType} recorder state: ${recorder?.state || 'null'}`)
          // Clear flag even if recorder is not active
          activeRecordingByTypeRef.current.set(violationType, false)
          violationRecordersRef.current.delete(violationType)
          violationChunksByTypeRef.current.delete(violationType)
          violationDetectionTimesRef.current.delete(violationType)
        }
        violationTimeoutsRef.current.delete(violationType)
      }, 10000) // 10 seconds after detection
      
      violationTimeoutsRef.current.set(violationType, timeoutId)
    } catch (error) {
      console.error(`❌ Error starting violation recording for ${violationType}:`, error)
      activeRecordingByTypeRef.current.set(violationType, false)
      violationRecordersRef.current.delete(violationType)
      violationChunksByTypeRef.current.delete(violationType)
      violationDetectionTimesRef.current.delete(violationType)
      const pid = violationPendingIdsRef.current.get(violationType)
      if (pid) {
        markRecordedVideoFailed(pid)
        violationPendingIdsRef.current.delete(violationType)
      }
    }
  }, [addLogEntry, getBestMimeType, addRecordedVideo, addPendingRecordedVideo, finalizeRecordedVideo, markRecordedVideoFailed, validateBlobDuration, sendVideoToAPI])


  // Add violation to list when detected
  const addViolation = useCallback((type: DetectionType, score?: number) => {
    // For face_not_visible, cell_phone, multiple_faces, and tab_switch, track as duration - update existing or create new
    if (type === 'face_not_visible' || type === 'cell_phone' || type === 'multiple_faces' || type === 'tab_switch' || type === 'wrong_face' || type === 'eyes_off_screen') {
      setViolations(prev => {
        const activeRef = type === 'face_not_visible'
          ? activeFaceNotVisibleViolationRef
          : type === 'cell_phone'
          ? activeCellPhoneViolationRef
          : type === 'multiple_faces'
          ? activeMultipleFacesViolationRef
          : type === 'tab_switch'
          ? activeTabSwitchViolationRef
          : type === 'wrong_face'
          ? activeWrongFaceViolationRef
          : activeEyesOffScreenViolationRef
        
        // Check if there's an active violation of this type
        if (activeRef.current !== null) {
          const activeIndex = activeRef.current
          // Verify the violation at this index is still the same type
          if (prev[activeIndex] && prev[activeIndex].type === type) {
            // Update the end time of the existing violation (maintain 10 seconds after current time)
            const updated = [...prev]
            const currentTime = new Date()
            updated[activeIndex] = {
              ...updated[activeIndex],
              endTime: new Date(currentTime.getTime() + 10000), // Update end time to current time + 10 seconds
              violationTime: updated[activeIndex].violationTime, // Keep original violation time
              score: score !== undefined ? score : updated[activeIndex].score // Update score if provided
            }
            return updated
          } else {
            // Active index is invalid, reset it
            activeRef.current = null
          }
        }
        
        // Create new violation
        const violationTime = new Date()
        const startTime = new Date(violationTime.getTime() - 10000) // -10 seconds
        const endTime = new Date(violationTime.getTime() + 10000) // +10 seconds

        const violation: ViolationEntry = {
          type,
          violationTime,
          startTime,
          endTime,
          score
        }

        const newViolations = [...prev, violation]
        // Set the active index to the new violation
        activeRef.current = newViolations.length - 1
        
        // Add log entry
        const violationMessage = type === 'face_not_visible'
          ? 'Face Not Visible'
          : type === 'cell_phone'
          ? `Possible Phone Usage (${((score || 0) * 100).toFixed(1)}%)`
          : type === 'multiple_faces'
          ? 'Multiple Faces Detected'
          : type === 'tab_switch'
          ? 'Tab Switch Detected'
          : type === 'wrong_face'
          ? 'Face Mismatch Detected'
          : type === 'eyes_off_screen'
          ? 'Possible Looking Away Violation'
          : 'Violation Detected'
        addLogEntry(violationMessage)

        // Bottom-left toast label — short form of the same violation, matching
        // the "Warning/Amaran : <count> : <label>" convention.
        const toastLabel = type === 'face_not_visible'
          ? 'Face Not Visible'
          : type === 'cell_phone'
          ? 'Phone Usage'
          : type === 'multiple_faces'
          ? 'Multiple Faces'
          : type === 'tab_switch'
          ? 'Tab Switch'
          : type === 'wrong_face'
          ? 'Face Mismatch'
          : type === 'eyes_off_screen'
          ? 'Looking Away'
          : 'Violation'
        pushWarningToast(type, toastLabel)

        return newViolations
      })
    } else {
      // For other violation types, create new entry as before
      const violationTime = new Date()
      const startTime = new Date(violationTime.getTime() - 10000) // -10 seconds
      const endTime = new Date(violationTime.getTime() + 10000) // +10 seconds

      const violation: ViolationEntry = {
        type,
        violationTime,
        startTime,
        endTime,
        score
      }

      setViolations(prev => [...prev, violation])
      
      // Add log entry
      const violationMessage = 'Violation Detected'
      addLogEntry(violationMessage)
      
    }
  }, [addLogEntry, pushWarningToast])

  const loadMediaPipeModels = async () => {
    try {
      // Initialize MediaPipe vision tasks
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      )
      
      // Load Face Detector with improved parameters
      const faceDetector = await FaceDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
          delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        minDetectionConfidence: 0.6, // Increased from 0.5 for better accuracy
        minSuppressionThreshold: 0.5 // Increased from 0.3 to prevent duplicate detections on same person
      })
      
      faceDetectorRef.current = faceDetector
      
      // Load Object Detector for smartphone detection
      const objectDetector = await ObjectDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
          delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        scoreThreshold: 0.42, // Set to 42% for smooth cell phone detection
        maxResults: 10 // Increased to see more detections
        // Removed categoryAllowlist to detect all objects and filter manually
      })
      
      objectDetectorRef.current = objectDetector

      // Load Face Landmarker for head-pose + eye-gaze (drives eyes_off_screen).
      // Outputs the 4x4 facial transformation matrix (head yaw/pitch) and ARKit
      // blendshapes (eyeLookIn/Out/Up/Down) so we can tell where the candidate is
      // looking without hand-rolling iris geometry.
      const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true
      })

      faceLandmarkerRef.current = faceLandmarker

      setModelsLoaded(true)

      // Start detection interval - 100ms for real-time detection (10 FPS)
      detectionIntervalRef.current = window.setInterval(() => {
        detectWithMediaPipe()
      }, 100)
    } catch (error) {
      console.error('Error loading MediaPipe models:', error)
    }
  }


  const detectWithMediaPipe = async () => {
    // Episode-gap sweep: close an episode (clearing its cooldown windows) ONLY
    // after the violation has been absent for longer than EPISODE_GAP_MS. This
    // runs every tick (~100ms) and covers every violation type — including the
    // event-driven ones (tab_switch, wrong_face) — so a brief dropout never
    // re-triggers a duplicate event/upload, while a genuine new occurrence after
    // the gap is treated fresh.
    const nowEpisodeSweep = Date.now()
    for (const [type, lastSeen] of lastSeenRef.current.entries()) {
      if (nowEpisodeSweep - lastSeen > EPISODE_GAP_MS) {
        resetEventDetectionCount(type)
        lastSeenRef.current.delete(type)
      }
    }

    // Guard against overlapping detection cycles. The interval re-fires every
    // 100ms, but a single cycle runs two heavy MediaPipe models back-to-back
    // and can exceed that budget. Without this guard, overlapping
    // detectForVideo calls throw on non-monotonic timestamps and the whole
    // frame is silently dropped.
    if (isDetectingRef.current) return
    isDetectingRef.current = true
    try {
    const video = getVideoElement()

    if (
      video &&
      video.readyState === 4 &&
      faceDetectorRef.current &&
      objectDetectorRef.current
    ) {
      const videoWidth = video.videoWidth
      const videoHeight = video.videoHeight

      video.width = videoWidth
      video.height = videoHeight

      if (canvasRef.current) {
        canvasRef.current.width = videoWidth
        canvasRef.current.height = videoHeight

        try {
          const startTimeMs = performance.now()

          // Run MediaPipe Face Detection
          const faceResults = faceDetectorRef.current.detectForVideo(video, startTimeMs)

          // Run MediaPipe Object Detection for smartphones
          const objectResults = objectDetectorRef.current.detectForVideo(video, startTimeMs)

          // Throttled frame-timing diagnostics: confirm whether the ~100ms
          // detection loop budget is actually being met (logged at most once
          // every ~3s). Informational only - no behavior change.
          const cycleMs = performance.now() - startTimeMs
          const nowForTiming = Date.now()
          if (nowForTiming - lastTimingLogRef.current >= 3000) {
            lastTimingLogRef.current = nowForTiming
            console.debug(
              `[proctoring] MediaPipe detection cycle: ${cycleMs.toFixed(1)}ms (budget 100ms)`
            )
          }

          const ctx = canvasRef.current.getContext('2d')
          if (ctx) {
            // Clear canvas
            ctx.clearRect(0, 0, videoWidth, videoHeight)

            const faceCount = faceResults.detections.length

            // Draw face detections (person detection overlay)
            if (showPersonOverlayRef.current)
            faceResults.detections.forEach((detection, index) => {
              const bbox = detection.boundingBox
              if (bbox) {
                const x = bbox.originX
                const y = bbox.originY
                const width = bbox.width
                const height = bbox.height
                
                // Different colors for single vs multiple faces
                const boxColor = faceCount > 1 ? '#ff0000' : '#00ff00' // Red if multiple, green if single
                
                // Draw bounding box
                ctx.strokeStyle = boxColor
                ctx.lineWidth = 3
                ctx.strokeRect(x, y, width, height)
                
                // Draw label background
                ctx.fillStyle = boxColor
                ctx.fillRect(x, y - 25, 180, 25)
                
                // Draw label text
                ctx.fillStyle = '#ffffff'
                ctx.font = '16px Arial'
                const confidence = detection.categories && detection.categories[0] 
                  ? (detection.categories[0].score * 100).toFixed(1) 
                  : '0.0'
                ctx.fillText(
                  `Face ${index + 1} (${confidence}%)`,
                  x + 5,
                  y - 8
                )
              }
            })

            // Draw object detections - ONLY cell phones (phone usage overlay)
            if (showPhoneOverlayRef.current && objectResults.detections && objectResults.detections.length > 0) {
              objectResults.detections.forEach((detection) => {
                if (!detection.categories || detection.categories.length === 0) return
                
                const category = detection.categories[0]
                const categoryName = category.categoryName || ''
                const categoryLower = categoryName.toLowerCase()
                
                // Check if this is a cell phone or phone-related object
                const isCellPhone = categoryLower.includes('cell phone') || 
                                   categoryLower.includes('phone') ||
                                   categoryLower.includes('mobile') ||
                                   categoryLower === 'cell phone'
                
                // ONLY draw and process cell phones with 42% or more confidence - ignore all other objects
                if (!isCellPhone || category.score < 0.42) return
                
                const bbox = detection.boundingBox
                if (bbox) {
                  const x = bbox.originX
                  const y = bbox.originY
                  const width = bbox.width
                  const height = bbox.height
                  
                  // Draw cell phone with magenta color
                  const boxColor = '#ff00ff' // Magenta for phones
                  
                  // Draw bounding box
                  ctx.strokeStyle = boxColor
                  ctx.lineWidth = 4
                  ctx.strokeRect(x, y, width, height)
                  
                  // Draw label background
                  ctx.fillStyle = boxColor
                  const labelWidth = Math.max(200, categoryName.length * 10)
                  ctx.fillRect(x, y - 25, labelWidth, 25)
                  
                  // Draw label text
                  ctx.fillStyle = '#ffffff'
                  ctx.font = 'bold 16px Arial'
                  const confidence = (category.score * 100).toFixed(1)
                  ctx.fillText(
                    `${categoryName} (${confidence}%)`,
                    x + 5,
                    y - 8
                  )
                }
              })
            }

            // Display detection status at top of canvas (person detection overlay)
            if (showPersonOverlayRef.current && faceCount > 1) {
              ctx.fillStyle = '#ff0000'
              ctx.fillRect(10, 10, 220, 40)
              ctx.fillStyle = '#ffffff'
              ctx.font = 'bold 18px Arial'
              ctx.fillText(
                `Multiple Faces: ${faceCount}`,
                15,
                35
              )
            } else if (showPersonOverlayRef.current && faceCount === 1) {
              ctx.fillStyle = '#00ff00'
              ctx.fillRect(10, 10, 200, 40)
              ctx.fillStyle = '#ffffff'
              ctx.font = 'bold 18px Arial'
              ctx.fillText(
                `Face Detected`,
                15,
                35
              )
            }

          // Detection thresholds for MediaPipe
          const REQUIRED_CONSECUTIVE_DETECTIONS_CELL_PHONE = 2 // 2 frames (0.2 seconds) - smooth detection with higher confidence
          const CELL_PHONE_DEBOUNCE_MS = 800 // 0.8 seconds debounce for smoother detection
          // Time-based gating for face-count violations (frame-count gating was
          // unreliable when the loop ran slower than 100ms/frame). Near-instant
          // alerting per requirement; raise to 500-800ms if too noisy.
          const FACE_NOT_VISIBLE_THRESHOLD_MS = 150
          const MULTIPLE_FACES_THRESHOLD_MS = 150
          // eyes_off_screen gating + thresholds. We only want to record GROSS
          // look-aways (clearly turning the head/eyes away from the screen), not
          // the small natural glances candidates make constantly. So the limits
          // are deliberately high and the look-away must be SUSTAINED (~2s)
          // before firing. Head yaw/pitch are in degrees; gaze magnitudes are
          // ARKit blendshape scores (0..1). A head turned/tilted away (left, right,
          // up OR down) with the iris FOLLOWING it counts as looking away.
          // Sustained-hold durations before a look-away is recorded. Horizontal
          // (left/right) and vertical (up/down) are tuned separately: glancing
          // down at a keyboard/desk is more common and natural than a sustained
          // side glance, so vertical is held longer to avoid false positives.
          const EYES_OFF_SCREEN_HORIZONTAL_MS = 1500 // left/right sustained hold
          // Vertical holds are DIRECTION-SPLIT like the limits: browser chrome
          // (URL bar, downloads) lives at the top of the screen, so up gets a
          // long hold; down is the prime cheat direction (desk notes, phone in
          // lap), so it fires fast. NOTE: sustained keyboard-gazing while
          // typing can resemble reading desk notes — if that false-fires in
          // practice, raise EYES_OFF_SCREEN_DOWN_MS rather than the limits.
          const EYES_OFF_SCREEN_UP_MS = 2000 // up sustained hold (lenient)
          const EYES_OFF_SCREEN_DOWN_MS = 800 // down sustained hold (fast)
          // Graduated hold: the 1500ms holds above exist to filter BORDERLINE
          // glances (score hovering near 1.0). A GROSS look-away — score well
          // past threshold, e.g. eyes dropped to a phone in the lap (~1.5-2.0)
          // — is unambiguous, so waiting the full hold only delays the alert.
          const GROSS_LOOK_AWAY_SCORE = 1.4 // score at/past this = clearly away
          const EYES_OFF_SCREEN_GROSS_MS = 600 // fast hold for gross look-aways
          const HEAD_YAW_LIMIT_DEG = 30 // clear left/right head turn
          // Gaze that, on its own, equals ONE full unit of look-away. Combined
          // with head pose below, so these are the point where the EYES alone
          // (head straight) count as off-screen. Lower = more sensitive. Kept low
          // enough that a moderate eye-drop to read desk notes contributes even
          // before the iris reaches the extreme edge of the eye.
          // gazeH sums two blendshapes per direction (range ~ -2..2); gazeV takes a
          // single-eye max (range ~ -1..1) — hence the different limits.
          const GAZE_HORIZONTAL_LOOK_AWAY = 0.6 // eye-only left/right swivel = 1 unit
          // Vertical is split by DIRECTION. Cheating on desk notes is a
          // downward glance, so DOWN stays sensitive. UP is given much more
          // headroom: on a large/tall monitor, glancing at the URL bar or the
          // top of the page is a small head-up + eye-up that is still
          // on-screen, and the old symmetric limits summed those into a false
          // violation.
          const HEAD_PITCH_UP_LIMIT_DEG = 50 // head tilt UP that alone = look-away
          const HEAD_PITCH_DOWN_LIMIT_DEG = 20 // head tilt DOWN that alone = look-away
          const GAZE_UP_LOOK_AWAY = 1.0 // eye-only upward swivel = 1 unit (lenient)
          // Eye-only down is deliberately NOT enough to fire from a normal
          // screen glance: the bottom of the screen (submit buttons!) is an
          // eye-drop of ~0.3-0.5, while reading the desk needs either an
          // extreme eye-drop or real head-down. Head-down + eyes-down together
          // (the actual notes/phone posture) still lands well past 1.0.
          const GAZE_DOWN_LOOK_AWAY = 0.65
          // Both lids at/past this blink score = eyes closed. Gaze blendshapes
          // are lid-driven and spike toward "down" during blinks/closure, so
          // gaze is unscorable then — and closed eyes can't read notes anyway.
          // Kept strict (0.7) because a steep down-gaze also droops the lids
          // (~0.3-0.5) and must NOT be mistaken for closure.
          const EYES_CLOSED_BLINK_SCORE = 0.7
          // Flicker tolerance. A turned head often makes the face detector or
          // the per-frame scores flicker for a frame or two; without a grace
          // window those flickers restart the sustained-hold clock and a real
          // look-away never accumulates its 1.5s. The hold only resets after
          // the candidate has read as on-screen (or face-lost) CONTINUOUSLY
          // for this long.
          const EYES_ON_RESET_GRACE_MS = 500
          // The posture baseline is only dropped (forcing recalibration) after
          // the single-face view has been gone this long — i.e. the candidate
          // genuinely left or repositioned, not a 1-frame detector dropout.
          const BASELINE_RESET_AFTER_NO_FACE_MS = 10000
          // When seeding the baseline we clamp toward neutral so that a
          // look-away pose at seed time can't become the "on-screen" reference
          // (which would suppress — or worse, invert — detection). Legit
          // camera-geometry offsets (laptop cam below screen, etc.) fall well
          // inside these clamps, and the EMA refines the rest once settled.
          const BASELINE_SEED_YAW_CLAMP_DEG = 15
          const BASELINE_SEED_PITCH_CLAMP_DEG = 15
          const BASELINE_SEED_GAZE_H_CLAMP = 0.3
          const BASELINE_SEED_GAZE_V_CLAMP = 0.25

          let currentDetection: DetectionType = null
          
          // Detect smartphone using object detection
          let isCellPhoneDetected = false
          let cellPhoneScore = 0
          
          if (objectResults.detections && objectResults.detections.length > 0) {
            // Only check for cell phones - filter out all other objects
            objectResults.detections.forEach((detection) => {
              if (detection.categories && detection.categories.length > 0) {
                const category = detection.categories[0]
                const categoryLower = category.categoryName.toLowerCase()
                
                // Check for cell phone category (comprehensive list of possible names)
                // COCO dataset uses "cell phone" as the category name
                const isPhone = categoryLower === 'cell phone' ||
                               categoryLower.includes('cell phone') || 
                               categoryLower.includes('cellphone') ||
                               categoryLower === 'phone' ||
                               categoryLower === 'mobile' ||
                               categoryLower === 'mobile phone' ||
                               categoryLower === 'smartphone'
                
                // Only consider detections with 42% or more confidence for smooth detection
                if (isPhone && category.score >= 0.42) {
                  isCellPhoneDetected = true
                  cellPhoneScore = Math.max(cellPhoneScore, category.score)
                  
                }
              }
            })
          }
          
          // Keep the cell-phone episode alive while the phone is visible.
          if (isCellPhoneDetected) lastSeenRef.current.set('cell_phone', Date.now())

          // Set cell phone detection if phone detected
          if (isCellPhoneDetected && !currentDetection) {
            currentDetection = 'cell_phone'
            latestDetectionScoreRef.current = { type: 'cell_phone', score: cellPhoneScore }
            
            // Add to detection history
            setDetectionHistory(prev => {
              const updated = [
                ...prev,
                {
                  type: 'cell_phone' as DetectionType,
                  timestamp: new Date(),
                  score: cellPhoneScore
                }
              ]
              return updated.slice(-100)
            })
          } else if (!isCellPhoneDetected && activeCellPhoneViolationRef.current !== null) {
            // End cell phone violation if no longer detected (add 10 seconds after)
            setViolations(prev => {
              const activeIndex = activeCellPhoneViolationRef.current
              if (activeIndex !== null && prev[activeIndex] && prev[activeIndex].type === 'cell_phone') {
                const updated = [...prev]
                const currentTime = new Date()
                updated[activeIndex] = {
                  ...updated[activeIndex],
                  endTime: new Date(currentTime.getTime() + 10000) // +10 seconds after detection ends
                }
                
                // Violation recording stops automatically after 10 seconds from detection
                // No need to schedule stop here
                
                return updated
              }
              return prev
            })
            activeCellPhoneViolationRef.current = null
            // Episode close (resetEventDetectionCount) is deferred to the
            // EPISODE_GAP_MS sweep so a brief dropout doesn't re-fire event/upload.
          }

          // Check for multiple faces (time-based gating, fired directly)
          if (faceCount > 1) {
            lastSeenRef.current.set('multiple_faces', Date.now())
            if (multipleFacesSinceRef.current === null) multipleFacesSinceRef.current = Date.now()

            if (Date.now() - multipleFacesSinceRef.current >= MULTIPLE_FACES_THRESHOLD_MS && !currentDetection) {
              currentDetection = 'multiple_faces'
              latestDetectionScoreRef.current = { type: 'multiple_faces', score: faceCount }

              // Fire directly, bypassing the consecutive-frame debounce
              const streamToUse = getMediaStream()
              const examIsActive = isExamActiveRef.current || isExamActive
              if (examIsActive && streamToUse) {
                startViolationRecording(streamToUse, new Date(), 'multiple_faces').catch((error) => {
                  console.error('Error starting violation recording:', error)
                })
              }
              addViolation('multiple_faces', faceCount)
            }
          } else {
            multipleFacesSinceRef.current = null
            if (activeMultipleFacesViolationRef.current !== null) {
              // End multiple faces violation (add 10 seconds after)
              setViolations(prev => {
                const activeIndex = activeMultipleFacesViolationRef.current
                if (activeIndex !== null && prev[activeIndex] && prev[activeIndex].type === 'multiple_faces') {
                  const updated = [...prev]
                  const currentTime = new Date()
                  updated[activeIndex] = {
                    ...updated[activeIndex],
                    endTime: new Date(currentTime.getTime() + 10000) // +10 seconds after detection ends
                  }
                  return updated
                }
                return prev
              })
              activeMultipleFacesViolationRef.current = null
              // Episode close deferred to the EPISODE_GAP_MS sweep.
            }
          }

          // Check for face not visible (time-based; only if nothing higher-priority detected)
          if (faceCount === 0 && !currentDetection) {
            lastSeenRef.current.set('face_not_visible', Date.now())
            if (faceNotVisibleSinceRef.current === null) faceNotVisibleSinceRef.current = Date.now()

            if (Date.now() - faceNotVisibleSinceRef.current >= FACE_NOT_VISIBLE_THRESHOLD_MS) {
              currentDetection = 'face_not_visible'
              latestDetectionScoreRef.current = { type: 'face_not_visible', score: 0 }

              // Fire directly, bypassing the consecutive-frame debounce
              const streamToUse = getMediaStream()
              const examIsActive = isExamActiveRef.current || isExamActive
              if (examIsActive && streamToUse) {
                startViolationRecording(streamToUse, new Date(), 'face_not_visible').catch((error) => {
                  console.error('Error starting violation recording:', error)
                })
              }
              addViolation('face_not_visible', 0)
            }
          } else if (faceCount > 0) {
            faceNotVisibleSinceRef.current = null
            if (activeFaceNotVisibleViolationRef.current !== null) {
              setViolations(prev => {
                const activeIndex = activeFaceNotVisibleViolationRef.current
                if (activeIndex !== null && prev[activeIndex] && prev[activeIndex].type === 'face_not_visible') {
                  const updated = [...prev]
                  const currentTime = new Date()
                  updated[activeIndex] = {
                    ...updated[activeIndex],
                    endTime: new Date(currentTime.getTime() + 10000) // +10 seconds after detection ends
                  }
                  return updated
                }
                return prev
              })
              activeFaceNotVisibleViolationRef.current = null
              // Episode close deferred to the EPISODE_GAP_MS sweep.
            }
          }

          // --- eyes_off_screen: head-pose + eye-gaze (only when exactly ONE face) ---
          // The FaceLandmarker runs only for the normal single-face case, so it
          // adds no per-frame cost while a higher-priority face-count violation is
          // active, and the !currentDetection guard keeps it below cell_phone.
          if (faceCount === 1) noSingleFaceSinceRef.current = null
          if (faceCount === 1 && !currentDetection && faceLandmarkerRef.current) {
            let isEyesOff = false
            let offMagnitude = 0
            let requiredHoldMs = EYES_OFF_SCREEN_HORIZONTAL_MS
            // HUD readouts drawn on the overlay as visual proof of the decision.
            let hudYaw = 0
            let hudPitch = 0
            let hudGaze = 0
            let hudGazeV = 0
            try {
              const lmResult = faceLandmarkerRef.current.detectForVideo(video, startTimeMs)

              // Head pose from the 4x4 facial transformation matrix (column-major:
              // R[row][col] === data[col*4 + row]). Standard Tait-Bryan extraction.
              // Signed: +yaw = head turned to the subject's right, +pitch = up.
              let headYaw = 0
              let headPitch = 0
              const matrix = lmResult.facialTransformationMatrixes?.[0]?.data
              if (matrix) {
                headYaw = Math.atan2(matrix[2], Math.sqrt(matrix[6] * matrix[6] + matrix[10] * matrix[10])) * (180 / Math.PI) // left/right turn
                headPitch = Math.atan2(-matrix[6], matrix[10]) * (180 / Math.PI) // up(+)/down(-); flip sign here if inverted on your camera
              }

              // Eye gaze from ARKit blendshapes, kept SIGNED on both axes.
              // +gazeH = iris to the subject's right, -gazeH = left, ~0 = centered.
              // +gazeV = iris up, -gazeV = down, ~0 = centered.
              let gazeH = 0
              let gazeV = 0
              let eyesClosed = false
              const shapes = lmResult.faceBlendshapes?.[0]?.categories
              if (shapes) {
                const shape = (name: string) => shapes.find(c => c.categoryName === name)?.score ?? 0
                // +gazeH = iris to the subject's RIGHT. For the RIGHT eye that is
                // "out" (away from nose); for the LEFT eye that is "in" (toward
                // nose). The mirror-image applies for looking left. (These were
                // previously swapped, which inverted gazeH so it cancelled headYaw
                // in the signed score and suppressed real horizontal look-aways.)
                const gazeRight = shape('eyeLookOutRight') + shape('eyeLookInLeft')
                const gazeLeft = shape('eyeLookOutLeft') + shape('eyeLookInRight')
                gazeH = gazeRight - gazeLeft
                const gazeUp = Math.max(shape('eyeLookUpLeft'), shape('eyeLookUpRight'))
                const gazeDown = Math.max(shape('eyeLookDownLeft'), shape('eyeLookDownRight'))
                gazeV = gazeUp - gazeDown
                // min of both lids: a one-eyed wink must not disable scoring.
                eyesClosed = Math.min(shape('eyeBlinkLeft'), shape('eyeBlinkRight')) >= EYES_CLOSED_BLINK_SCORE
              }

              // Combined head + eye look-away score, one per axis. Head pose and
              // gaze are each normalized to their own limit, then merged so that a
              // MODERATE head movement plus a MODERATE eye movement in the same
              // direction add up to a look-away — even though neither crossed its
              // threshold alone. This is the "looking down to read desk notes"
              // case: the head dips a little and the eyes roll down a little, and
              // together that is clearly off-screen. A score magnitude >= 1 = away.
              //
              // Horizontal uses a SIGNED sum so that eyes counter-rotating against a
              // turned head (to keep the screen in view) cancel out and stay ON
              // screen. Vertical uses MAGNITUDES (sign-agnostic) because the head
              // pitch sign can be camera-dependent, and for reading notes the head
              // and eyes both go down together anyway.
              // Score look-away as DEVIATION from the resting baseline instead of
              // absolute angles. Seed on the first frame; afterwards the baseline
              // EMA-tracks only while the candidate looks settled/on-screen (see
              // the update at the bottom of this block), so it captures the
              // camera geometry + natural posture — e.g. a laptop camera below
              // the screen makes "looking at the screen" a downward gaze, and the
              // baseline absorbs that so it no longer reads as a look-away.
              const baseline = gazeBaselineRef.current
              const dYaw = headYaw - (baseline?.yaw ?? headYaw)
              const dPitch = headPitch - (baseline?.pitch ?? headPitch)
              let dGazeH = gazeH - (baseline?.gazeH ?? gazeH)
              let dGazeV = gazeV - (baseline?.gazeV ?? gazeV)
              // Closed lids make the gaze estimates garbage (they read as a
              // hard down-gaze), and closed eyes can't read notes — so drop
              // the gaze terms entirely. Head-pose scoring stays live: a full
              // head turn/dip is a look-away regardless of lid state.
              if (eyesClosed) {
                dGazeH = 0
                dGazeV = 0
              }

              const hScore = dYaw / HEAD_YAW_LIMIT_DEG + dGazeH / GAZE_HORIZONTAL_LOOK_AWAY

              // Learn the camera's pitch-sign convention from head/eye
              // agreement: during a vertical glance the head and the eyes move
              // the SAME direction, and the eye-gaze sign (blendshapes) is
              // anatomically fixed. If accumulated evidence says head pitch
              // disagrees, the matrix pitch is inverted on this camera — flip
              // it. Without this, an inverted camera scores upward glances
              // with the strict DOWN limits and downward glances with the
              // lenient UP limits (i.e. up over-fires, down under-fires).
              if (Math.abs(dGazeV) > 0.15 && Math.abs(dPitch) > 3) {
                const agree = Math.sign(dPitch) === Math.sign(dGazeV) ? 1 : -1
                pitchSignEvidenceRef.current = Math.max(-50, Math.min(50, pitchSignEvidenceRef.current + agree))
              }
              const dPitchV = (pitchSignEvidenceRef.current < 0 ? -1 : 1) * dPitch

              // Vertical: score UP and DOWN separately (with direction-specific
              // limits) and take the worse one. Only same-direction head+eye
              // components add — head-down + eyes-up (slouching while still
              // reading the screen) no longer sums into a false look-away the
              // way the old |dPitch|+|dGazeV| form did. +dPitchV/+dGazeV = up.
              const vScoreUp =
                Math.max(0, dPitchV) / HEAD_PITCH_UP_LIMIT_DEG + Math.max(0, dGazeV) / GAZE_UP_LOOK_AWAY
              const vScoreDown =
                Math.max(0, -dPitchV) / HEAD_PITCH_DOWN_LIMIT_DEG + Math.max(0, -dGazeV) / GAZE_DOWN_LOOK_AWAY
              const vScore = Math.max(vScoreUp, vScoreDown)

              // A CLEAR head turn (relative to baseline) on its own is off-screen,
              // even if the gaze blendshapes disagree. At an extreme left/right
              // profile one eye is occluded and MediaPipe's gaze estimate goes
              // unreliable (often reporting the opposite direction), which would
              // otherwise cancel the large yaw in the signed hScore and suppress
              // the detection.
              const grossHeadTurn = Math.abs(dYaw) >= HEAD_YAW_LIMIT_DEG
              const horizontalLookAway = Math.abs(hScore) >= 1 || grossHeadTurn
              const upLookAway = vScoreUp >= 1
              const downLookAway = vScoreDown >= 1
              const verticalLookAway = upLookAway || downLookAway

              // Seed the baseline on the first landmarked frame, CLAMPED toward
              // neutral. With no baseline all deviations read 0, so without the
              // clamp whatever pose the candidate happened to hold at seed time
              // (including mid-look-away, e.g. right after a face-detector
              // dropout) became the "on-screen" reference — suppressing real
              // look-aways and flagging the return to the screen instead.
              if (!baseline) {
                const clamp = (v: number, lim: number) => Math.max(-lim, Math.min(lim, v))
                gazeBaselineRef.current = {
                  yaw: clamp(headYaw, BASELINE_SEED_YAW_CLAMP_DEG),
                  pitch: clamp(headPitch, BASELINE_SEED_PITCH_CLAMP_DEG),
                  gazeH: clamp(gazeH, BASELINE_SEED_GAZE_H_CLAMP),
                  gazeV: clamp(gazeV, BASELINE_SEED_GAZE_V_CLAMP)
                }
              } else if (Math.abs(hScore) < 0.5 && vScore < 0.5 && !eyesClosed) {
                // Adapt the baseline toward the current posture ONLY when the
                // candidate looks settled on-screen (both axes well under
                // threshold) with eyes open — closed-lid frames carry garbage
                // gaze values that would train the baseline toward "down".
                // Slow EMA so a brief centered glance can't reset it, and
                // excluding flagged frames means a sustained look-away never
                // trains the baseline to accept itself.
                const a = 0.03 // EMA weight
                baseline.yaw += a * (headYaw - baseline.yaw)
                baseline.pitch += a * (headPitch - baseline.pitch)
                baseline.gazeH += a * (gazeH - baseline.gazeH)
                baseline.gazeV += a * (gazeV - baseline.gazeV)
              }

              // HUD shows the DEVIATION values that actually drive the scores
              // (pitch shown sign-corrected, since that's what gets scored).
              hudYaw = dYaw
              hudPitch = dPitchV
              hudGaze = dGazeH
              hudGazeV = dGazeV
              if (horizontalLookAway || verticalLookAway) {
                isEyesOff = true
                // When several directions are off at once, fire on the
                // earliest (shortest) applicable hold.
                requiredHoldMs = Math.min(
                  horizontalLookAway ? EYES_OFF_SCREEN_HORIZONTAL_MS : Infinity,
                  upLookAway ? EYES_OFF_SCREEN_UP_MS : Infinity,
                  downLookAway ? EYES_OFF_SCREEN_DOWN_MS : Infinity
                )
                offMagnitude = Math.max(
                  Math.abs(hScore),
                  grossHeadTurn ? Math.abs(dYaw) / HEAD_YAW_LIMIT_DEG : 0,
                  vScore
                )
                // Unambiguous look-away -> shorten the hold so blatant cases
                // (eyes on a phone in the lap, full head turn to notes) alert
                // quickly; borderline scores still ride out the full hold.
                if (offMagnitude >= GROSS_LOOK_AWAY_SCORE) {
                  requiredHoldMs = Math.min(requiredHoldMs, EYES_OFF_SCREEN_GROSS_MS)
                }
              }

              // --- Visual proof: draw the face mesh (skeleton) + eyes/irises and a
              // live HUD, colored red while off-screen and green while on-screen. ---
              const landmarks = lmResult.faceLandmarks?.[0]
              if (landmarks && showEyesOverlayRef.current) {
                const drawingUtils = new DrawingUtils(ctx)
                const meshColor = isEyesOff ? 'rgba(255,64,64,0.45)' : 'rgba(0,255,128,0.35)'
                const eyeColor = isEyesOff ? '#ff2d2d' : '#00ff88'

                // Full-face tessellation = the "skeleton" wireframe.
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: meshColor, lineWidth: 1 })
                // Highlight both eyes and irises so gaze is easy to read.
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: eyeColor, lineWidth: 2 })
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: eyeColor, lineWidth: 2 })
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: eyeColor, lineWidth: 2 })
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: eyeColor, lineWidth: 2 })

                // HUD readout box (top-right corner).
                const boxW = 250
                const boxX = videoWidth - boxW - 10
                ctx.fillStyle = isEyesOff ? 'rgba(200,0,0,0.75)' : 'rgba(0,140,60,0.75)'
                ctx.fillRect(boxX, 10, boxW, 122)
                ctx.fillStyle = '#ffffff'
                ctx.font = 'bold 16px Arial'
                ctx.fillText(isEyesOff ? 'EYES OFF SCREEN' : eyesClosed ? 'EYES CLOSED' : 'EYES ON SCREEN', boxX + 10, 32)
                ctx.font = '13px Arial'
                ctx.fillText(`yaw ${hudYaw.toFixed(0)}°/${HEAD_YAW_LIMIT_DEG}°  pitch ${hudPitch.toFixed(0)}° (up/${HEAD_PITCH_UP_LIMIT_DEG}° dn/${HEAD_PITCH_DOWN_LIMIT_DEG}°)`, boxX + 10, 54)
                ctx.fillText(`irisH ${hudGaze.toFixed(2)}  irisV ${hudGazeV.toFixed(2)}`, boxX + 10, 72)
                ctx.fillText(`H-score ${Math.abs(hScore).toFixed(2)}/1.0 (head+eyes)`, boxX + 10, 90)
                ctx.fillText(`V-score ${vScore.toFixed(2)}/1.0 (up ${vScoreUp.toFixed(2)} / dn ${vScoreDown.toFixed(2)})`, boxX + 10, 108)
                ctx.fillText(`held ${eyesOffScreenSinceRef.current ? (Date.now() - eyesOffScreenSinceRef.current) : 0}ms / ${requiredHoldMs}ms`, boxX + 10, 126)
              }
            } catch (lmError) {
              console.error('Error during FaceLandmarker detection:', lmError)
            }

            if (isEyesOff) {
              eyesOnGraceStartRef.current = null
              lastSeenRef.current.set('eyes_off_screen', Date.now())
              if (eyesOffScreenSinceRef.current === null) eyesOffScreenSinceRef.current = Date.now()

              if (Date.now() - eyesOffScreenSinceRef.current >= requiredHoldMs) {
                currentDetection = 'eyes_off_screen'
                latestDetectionScoreRef.current = { type: 'eyes_off_screen', score: offMagnitude }

                // Fire directly, mirroring multiple_faces / face_not_visible.
                const streamToUse = getMediaStream()
                const examIsActive = isExamActiveRef.current || isExamActive
                if (examIsActive && streamToUse) {
                  startViolationRecording(streamToUse, new Date(), 'eyes_off_screen').catch((error) => {
                    console.error('Error starting violation recording:', error)
                  })
                }
                addViolation('eyes_off_screen', offMagnitude)
              }
            } else {
              // Don't restart the sustained-hold clock on a single noisy
              // on-screen frame — require the on-screen reading to persist for
              // the grace window first.
              if (eyesOnGraceStartRef.current === null) eyesOnGraceStartRef.current = Date.now()
              if (Date.now() - eyesOnGraceStartRef.current >= EYES_ON_RESET_GRACE_MS) {
                eyesOffScreenSinceRef.current = null
              }
              if (activeEyesOffScreenViolationRef.current !== null) {
                setViolations(prev => {
                  const activeIndex = activeEyesOffScreenViolationRef.current
                  if (activeIndex !== null && prev[activeIndex] && prev[activeIndex].type === 'eyes_off_screen') {
                    const updated = [...prev]
                    const currentTime = new Date()
                    updated[activeIndex] = {
                      ...updated[activeIndex],
                      endTime: new Date(currentTime.getTime() + 10000) // +10 seconds after detection ends
                    }
                    return updated
                  }
                  return prev
                })
                activeEyesOffScreenViolationRef.current = null
                // Episode close deferred to the EPISODE_GAP_MS sweep.
              }
            }
          } else if (faceCount !== 1) {
            // A turned head frequently makes the face detector drop the face
            // for a frame or two, so treat face-count flicker with the same
            // grace as an on-screen flicker: the look-away hold only resets
            // after the grace window, and the posture baseline is only dropped
            // (for recalibration) once the single-face view has been gone long
            // enough to mean the candidate actually left or repositioned —
            // otherwise it would re-seed from whatever pose they hold when the
            // face returns, poisoning the reference.
            if (noSingleFaceSinceRef.current === null) noSingleFaceSinceRef.current = Date.now()
            if (eyesOnGraceStartRef.current === null) eyesOnGraceStartRef.current = Date.now()
            if (Date.now() - eyesOnGraceStartRef.current >= EYES_ON_RESET_GRACE_MS) {
              eyesOffScreenSinceRef.current = null
            }
            if (Date.now() - noSingleFaceSinceRef.current >= BASELINE_RESET_AFTER_NO_FACE_MS) {
              gazeBaselineRef.current = null
            }
          }

          // Track consecutive cell-phone detections to reduce false positives.
          // multiple_faces / face_not_visible are now fired directly above and
          // no longer flow through this consecutive-frame debounce.
          const cellPhoneDetection: DetectionType = currentDetection === 'cell_phone' ? 'cell_phone' : null
          if (cellPhoneDetection === detectionCountRef.current.type) {
            detectionCountRef.current.count++
          } else {
            // End active cell-phone violation if switching away (add 10 seconds after)
            if (detectionCountRef.current.type === 'cell_phone' && activeCellPhoneViolationRef.current !== null) {
              setViolations(prev => {
                const activeIndex = activeCellPhoneViolationRef.current
                if (activeIndex !== null && prev[activeIndex] && prev[activeIndex].type === 'cell_phone') {
                  const updated = [...prev]
                  const currentTime = new Date()
                  updated[activeIndex] = {
                    ...updated[activeIndex],
                    endTime: new Date(currentTime.getTime() + 10000) // +10 seconds after detection ends
                  }
                  return updated
                }
                return prev
              })
              activeCellPhoneViolationRef.current = null
              // Episode close deferred to the EPISODE_GAP_MS sweep.
            }

            detectionCountRef.current.type = cellPhoneDetection
            detectionCountRef.current.count = cellPhoneDetection ? 1 : 0
          }

          // Only trigger the cell-phone violation after required consecutive detections
          const confirmedDetection = detectionCountRef.current.type
          const hasEnoughConsecutiveDetections =
            detectionCountRef.current.count >= REQUIRED_CONSECUTIVE_DETECTIONS_CELL_PHONE &&
            confirmedDetection === 'cell_phone'

          // For cell phones, also check debounce
          let shouldTriggerAlert: boolean = hasEnoughConsecutiveDetections
          if (confirmedDetection === 'cell_phone' && hasEnoughConsecutiveDetections) {
            const now = Date.now()
            const timeSinceLastDetection = now - lastCellPhoneDetectionTimeRef.current
            shouldTriggerAlert = timeSinceLastDetection >= CELL_PHONE_DEBOUNCE_MS
          }

          if (shouldTriggerAlert && confirmedDetection === 'cell_phone') {
            // Start violation recording automatically (will be ignored if already recording this type)
            const streamToUse = getMediaStream()
            const examIsActive = isExamActiveRef.current || isExamActive // Check both ref and state

            if (examIsActive && streamToUse) {
              const detectionTime = new Date()
              // Fire and forget - violation recording runs in background
              // startViolationRecording will ignore if same type is already recording
              startViolationRecording(streamToUse, detectionTime, 'cell_phone').catch((error) => {
                console.error('Error starting violation recording:', error)
              })
            }
            lastCellPhoneDetectionTimeRef.current = Date.now()

            const violationScore = latestDetectionScoreRef.current?.type === 'cell_phone'
              ? latestDetectionScoreRef.current.score
              : undefined
            addViolation('cell_phone', violationScore)
          } else if (confirmedDetection === 'cell_phone' && activeCellPhoneViolationRef.current !== null) {
            setViolations(prev => {
              const activeIndex = activeCellPhoneViolationRef.current
              if (activeIndex !== null && prev[activeIndex] && prev[activeIndex].type === 'cell_phone') {
                const updated = [...prev]
                const currentTime = new Date()
                updated[activeIndex] = {
                  ...updated[activeIndex],
                  endTime: new Date(currentTime.getTime() + 10000), // +10 seconds after detection ends
                  score: latestDetectionScoreRef.current?.type === 'cell_phone'
                    ? latestDetectionScoreRef.current.score
                    : updated[activeIndex].score
                }
                return updated
              }
              return prev
            })
          }
          } // Close ctx if statement
        } catch (error) {
          console.error('Error during MediaPipe detection:', error)
        }
      } // Close canvasRef.current if statement
    }
    } finally {
      isDetectingRef.current = false
    }
  }

  // Track tab visibility for tab switch detection
  useEffect(() => {
    const handleVisibilityChange = () => {
      // Detect when user switches away from the tab (tab becomes hidden)
      if (document.hidden && isExamActiveRef.current) {
        // Trigger tab switch violation
        const detectionTime = new Date()
        detectionCountRef.current = { type: 'tab_switch', count: 1 }
        latestDetectionScoreRef.current = { type: 'tab_switch', score: 1 }
        lastSeenRef.current.set('tab_switch', Date.now())
        
        // Add violation
        addViolation('tab_switch')
        
        // Start violation recording if exam is active
        const streamToUse = getMediaStream()
        if (streamToUse) {
          startViolationRecording(streamToUse, detectionTime, 'tab_switch').catch((error) => {
            console.error('Error starting tab switch violation recording:', error)
          })
        }
      } else if (!document.hidden && activeTabSwitchViolationRef.current !== null) {
        // Tab is visible again - end the violation
        setViolations(prev => {
          const activeIndex = activeTabSwitchViolationRef.current
          if (activeIndex !== null && prev[activeIndex] && prev[activeIndex].type === 'tab_switch') {
            const updated = [...prev]
            const currentTime = new Date()
            updated[activeIndex] = {
              ...updated[activeIndex],
              endTime: new Date(currentTime.getTime() + 10000) // +10 seconds after tab becomes visible
            }
            return updated
          }
          return prev
        })
        activeTabSwitchViolationRef.current = null
        detectionCountRef.current = { type: null, count: 0 }
        // Episode close deferred to the EPISODE_GAP_MS sweep.
      }
    }

    const handleBlur = () => {
      // Window blur can also indicate tab switch
      if (isExamActiveRef.current && !document.hidden) {
        // Small delay to check if it's actually a tab switch
        setTimeout(() => {
          if (document.hidden && isExamActiveRef.current) {
            handleVisibilityChange()
          }
        }, 100)
      }
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)
    window.addEventListener('blur', handleBlur)

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      window.removeEventListener('blur', handleBlur)
    }
  }, [addViolation, getMediaStream, startViolationRecording])


  // Log detection history changes (for debugging/maintenance)
  useEffect(() => {
    // Removed console.log statements
  }, [detectionHistory])

  // Auto-start exam when webcam and models are both ready
  useEffect(() => {
    if (webcamReady && modelsLoaded && !hasAutoStartedRef.current) {
      hasAutoStartedRef.current = true
      handleStartExam()
    }
  }, [webcamReady, modelsLoaded, handleStartExam])

  // Auto-end exam when tab/window is closed (embedded scenario)
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (isExamActiveRef.current) {
        handleEndExam()
      }
    }
    window.addEventListener('beforeunload', handleBeforeUnload)
    return () => window.removeEventListener('beforeunload', handleBeforeUnload)
  }, [handleEndExam])

  // Extract session ID from URL parameters on mount
  useEffect(() => {
    const sid = extractSessionId()
    setProctorSessionId(sid)
  }, [extractSessionId])

  // Wrong-face proctoring: compares the live webcam feed against the KYC selfie.
  // Mismatches are routed through the exact same violation pipeline as every
  // other violation type — logged, recorded (10s clip), and uploaded/event-sent.
  const handleWrongFaceMismatch = useCallback((distance: number) => {
    // Only act while an exam is in progress (mirrors the other detections).
    if (!isExamActiveRef.current) return

    // Keep the wrong-face episode alive while the mismatch persists.
    lastSeenRef.current.set('wrong_face', Date.now())

    // Start the 10-second violation recording (deduped per type while active).
    const streamToUse = getMediaStream()
    if (streamToUse) {
      const detectionTime = new Date()
      startViolationRecording(streamToUse, detectionTime, 'wrong_face').catch((error) => {
        console.error('Error starting wrong face violation recording:', error)
      })
    }

    // Log + send event (addViolation throttles repeats and updates duration).
    addViolation('wrong_face', distance)
  }, [getMediaStream, startViolationRecording, addViolation])

  const handleWrongFaceMatch = useCallback(() => {
    if (activeWrongFaceViolationRef.current === null) return
    // Face matches again — close out the active violation (+10s like the others).
    setViolations(prev => {
      const activeIndex = activeWrongFaceViolationRef.current
      if (activeIndex !== null && prev[activeIndex] && prev[activeIndex].type === 'wrong_face') {
        const updated = [...prev]
        const currentTime = new Date()
        updated[activeIndex] = {
          ...updated[activeIndex],
          endTime: new Date(currentTime.getTime() + 10000), // +10 seconds after match resumes
        }
        return updated
      }
      return prev
    })
    activeWrongFaceViolationRef.current = null
    // Episode close deferred to the EPISODE_GAP_MS sweep.
  }, [])

  useFaceProctoring({
    sessionId: proctorSessionId,
    onMismatch: handleWrongFaceMismatch,
    onMatch: handleWrongFaceMatch,
    onStatus: setFaceProctoringStatus,
  })

  useEffect(() => {
    loadMediaPipeModels()

    // Cleanup on unmount
    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current)
        detectionIntervalRef.current = null
      }
      if (recordingTimeoutRef.current) {
        clearTimeout(recordingTimeoutRef.current)
        recordingTimeoutRef.current = null
      }
      if (cellPhoneDetectionDebounceRef.current) {
        clearTimeout(cellPhoneDetectionDebounceRef.current)
        cellPhoneDetectionDebounceRef.current = null
      }
      
      // Clean up MediaPipe detectors
      if (faceDetectorRef.current) {
        faceDetectorRef.current.close()
        faceDetectorRef.current = null
      }
      if (objectDetectorRef.current) {
        objectDetectorRef.current.close()
        objectDetectorRef.current = null
      }
      if (faceLandmarkerRef.current) {
        faceLandmarkerRef.current.close()
        faceLandmarkerRef.current = null
      }
      
      // Stop exam recording if active (using react-media-recorder)
      if (examRecordingStatus === 'recording') {
        stopExamRecording()
      }
      
      // Stop rolling buffer recorder if active
      if (rollingBufferRecorderRef.current && rollingBufferRecorderRef.current.state !== 'inactive') {
        rollingBufferRecorderRef.current.stop()
        rollingBufferRecorderRef.current = null
      }
      
      // Stop all active violation recorders
      for (const recorder of violationRecordersRef.current.values()) {
        if (recorder && recorder.state !== 'inactive') {
          recorder.stop()
        }
      }
      
      // Clear all violation timeouts
      for (const timeoutId of violationTimeoutsRef.current.values()) {
        if (timeoutId) {
          clearTimeout(timeoutId)
        }
      }
      
      // Clear all violation recording refs
      violationRecordersRef.current.clear()
      violationChunksByTypeRef.current.clear()
      violationDetectionTimesRef.current.clear()
      violationTimeoutsRef.current.clear()
      activeRecordingByTypeRef.current.clear()
      
      // Stop violation recorder if active (using react-media-recorder)
      if (violationRecordingStatus === 'recording') {
        stopViolationRecordingHook()
      }
      
      // Stop chunk recorder if active
      if (examChunkRecorderRef.current && examChunkRecorderRef.current.state !== 'inactive') {
        examChunkRecorderRef.current.stop()
        examChunkRecorderRef.current = null
      }
      
      // Clear cooldown windows on unmount
      lastEventSentRef.current.clear()
      lastVideoUploadRef.current.clear()
      lastSeenRef.current.clear()
      
      // Stop all media stream tracks
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop())
        mediaStreamRef.current = null
      }
    }
  }, [])

  // Calculate summary report stats (if needed in the future)
  // const faceNotCenteredCount = violations.filter(v => v.type === 'face_not_visible').length
  // const faceTooSmallCount = 0 // Placeholder
  // const noFaceDetectedCount = violations.filter(v => v.type === 'face_not_visible').length

  // Show eKYC flow if not completed
  // Temporarily hidden
  // if (!eKYCCompleted) {
  //   return <EKYC onComplete={() => setEKYCCompleted(true)} />
  // }

  return (
    // Downscale the entire app uniformly (layout + fonts) since it is embedded
    // in a small ~1/3 x 1/3 viewport the candidate never directly sees.
    // `zoom` is non-standard and silently ignored in some browsers/webviews
    // (so it "only worked on localhost"). `transform: scale()` is the
    // cross-browser equivalent, but it does not shrink the layout box, so we
    // widen/heighten the element by 1/scale (here 200% for scale 0.5) so the
    // scaled content still fills the viewport. Tune SCALE to taste.
    <div
      className="min-h-screen bg-transparent p-4"
      style={{
        transform: 'scale(0.5)',
        transformOrigin: 'top left',
        width: '200%',
        height: '200%',
      }}
    >
      <div className="max-w-7xl mx-auto space-y-4">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-3xl font-bold text-foreground">
            {' '}
          </h1>
        </div>

        {/* Control Buttons - Outside Cards */}
        <div className="flex items-center justify-center gap-3">
          <Button
            onClick={isExamActive ? handleEndExam : handleStartExam}
            disabled={!webcamReady || !modelsLoaded}
            variant={isExamActive ? 'destructive' : 'default'}
            size="lg"
            style={{ display: 'none' }}
            className="font-semibold"
          >
            {isExamActive ? 'TAMAT PEPERIKSAAN' : 'MULA PEPERIKSAAN'}
          </Button>
          <Button
            onClick={() => setIsViewVisible(!isViewVisible)}
            variant="ghost"
            size="icon"
            title={isViewVisible ? 'Hide view' : 'Show view'}
          >
            {isViewVisible ? (
              <Eye className="w-5 h-5" />
            ) : (
              <EyeOff className="w-5 h-5" />
            )}
          </Button>
        </div>

        {/* Video Feed and Log Section - Side by Side */}
        <div className={`grid grid-cols-1 lg:grid-cols-3 gap-4 ${!isViewVisible ? 'hidden' : ''}`}>
          {/* Video Feed - Takes 2 columns */}
          <Card className="lg:col-span-2 overflow-hidden">
            <CardContent className="p-0">
              <div className="relative w-full aspect-video bg-black overflow-hidden">
                {webcamError ? (
                  <div className="w-full h-full flex flex-col items-center justify-center text-white p-4">
                    <p className="text-lg font-semibold mb-2">Camera Access Error</p>
                    <p className="text-sm text-center">{webcamError}</p>
                    <p className="text-xs text-center mt-2 text-gray-300">
                      Please allow camera access and refresh the page.
                    </p>
                  </div>
                ) : (
                  <>
                    <Webcam
                      ref={webcamRef}
                      id="proctoring-video"
                      audio={false}
                      videoConstraints={{
                        width: { ideal: 854, max: 854 },
                        height: { ideal: 480, max: 480 },
                        facingMode: 'user',
                        frameRate: { ideal: 30, min: 24 },
                      }}
                      onUserMedia={handleUserMedia}
                      onUserMediaError={handleUserMediaError}
                      className="w-full h-full object-cover"
                      mirrored={false}
                    />
                    {!webcamReady && !webcamError && (
                      <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 text-white">
                        <p>Requesting camera access...</p>
                      </div>
                    )}
                  </>
                )}
                <canvas
                  ref={canvasRef}
                  className="absolute top-0 left-0 w-full h-full pointer-events-none"
                  style={{ zIndex: 10 }}
                />
                {/* Warning toaster — bottom-left, replaces the old on-canvas detection labels */}
                <div
                  className="absolute bottom-3 left-3 flex flex-col gap-2 pointer-events-none"
                  style={{ zIndex: 20 }}
                >
                  {warningToasts.map(t => (
                    <div
                      key={t.id}
                      className="flex items-center gap-2 rounded-md bg-white px-4 py-2.5 text-sm shadow-lg animate-in fade-in slide-in-from-left-2"
                    >
                      <span className="font-semibold text-slate-900">Warning/Amaran</span>
                      <span className="text-slate-400">:</span>
                      <span className="text-slate-700">{t.label}</span>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Log Section - Takes 1 column */}
          <div className="rounded-lg bg-[#f5f5f5] flex flex-col">
            <CardHeader className="flex-row items-center justify-between gap-2 space-y-0 py-3">
              <div className="flex items-center gap-2 flex-wrap">
                <CardTitle className="text-lg">Log Section</CardTitle>
                <Badge variant="secondary" className="font-normal">
                  <span className="font-semibold mr-1">Session ID:</span>
                  {sessionIdDisplay || 'No Session ID'}
                </Badge>
                {/* Wrong-face checker status badge hidden for now */}
                {false && (
                  <Badge
                    variant={
                      faceProctoringStatus.includes('DISABLED') || faceProctoringStatus.includes('FAILED')
                        ? 'destructive'
                        : 'secondary'
                    }
                    className="font-normal"
                    title="Live status of the wrong-face (KYC mismatch) checker"
                  >
                    {faceProctoringStatus}
                  </Badge>
                )}
              </div>
              <div className="relative" ref={overlayMenuRef}>
                <Button
                  onClick={() => setIsOverlayMenuOpen((open) => !open)}
                  variant={showPersonOverlay || showPhoneOverlay || showEyesOverlay ? 'default' : 'secondary'}
                  size="icon"
                  title="Toggle detection overlays"
                  aria-haspopup="true"
                  aria-expanded={isOverlayMenuOpen}
                >
                  <Layers className="w-5 h-5" />
                </Button>
                {isOverlayMenuOpen && (
                  <div className="absolute right-0 top-full z-50 mt-2 w-60 rounded-md border border-input bg-background p-1 shadow-md">
                    <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">
                      Show on webcam
                    </div>
                    {([
                      { label: 'Person detection', checked: showPersonOverlay, toggle: () => setShowPersonOverlay((v) => !v) },
                      { label: 'Phone usage detection', checked: showPhoneOverlay, toggle: () => setShowPhoneOverlay((v) => !v) },
                      { label: 'Eye looking away detection', checked: showEyesOverlay, toggle: () => setShowEyesOverlay((v) => !v) },
                    ] as const).map((item) => (
                      <button
                        key={item.label}
                        type="button"
                        onClick={item.toggle}
                        className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent hover:text-accent-foreground"
                      >
                        <span
                          className={cn(
                            'flex h-4 w-4 shrink-0 items-center justify-center rounded border',
                            item.checked
                              ? 'border-primary bg-primary text-primary-foreground'
                              : 'border-input bg-background'
                          )}
                        >
                          {item.checked && <Check className="h-3 w-3" />}
                        </span>
                        <span>{item.label}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </CardHeader>
            <CardContent className="p-4 flex-1 flex flex-col">
              <Textarea
                ref={logSectionRef}
                readOnly
                value={logEntries.join('\n')}
                className="h-full max-h-[300px] [field-sizing:fixed] font-mono text-xs overflow-y-auto resize-none"
                placeholder="Log entries will appear here..."
              />
            </CardContent>
          </div>
        </div>

        {/* Status Messages */}
        {!modelsLoaded && (
          <div className="text-center">
            <p className="text-primary font-semibold">Loading detection models...</p>
          </div>
        )}

        {/* Recording Timer - Inside show/hide section */}
        {isViewVisible && isExamActive && examStartTime && (
          <div className="rounded-lg bg-[#f5f5f5]">
            <CardContent className="p-4 text-center">
              <div className="flex items-center justify-center gap-3">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-destructive rounded-full animate-pulse"></div>
                  <p className="font-semibold text-lg">Recording in progress</p>
                </div>
                <div className="font-mono text-xl font-bold tabular-nums">
                  {Math.floor(recordingDuration / 3600).toString().padStart(2, '0')}:
                  {Math.floor((recordingDuration % 3600) / 60).toString().padStart(2, '0')}:
                  {(recordingDuration % 60).toString().padStart(2, '0')}
                </div>
              </div>
              <p className="text-sm text-muted-foreground mt-1">
                Started at {examStartTime.toLocaleTimeString()}
              </p>
            </CardContent>
          </div>
        )}

        {/* Recorded Videos List */}
        {isViewVisible && (
          <div className="rounded-lg bg-[#f5f5f5]">
            <CardHeader className="py-3">
              <CardTitle className="text-lg">Recorded Videos</CardTitle>
              <CardDescription>
                {recordedVideos.length > 0
                  ? `${recordedVideos.length} video${recordedVideos.length > 1 ? 's' : ''} recorded`
                  : 'No videos recorded yet'}
              </CardDescription>
            </CardHeader>
            <CardContent className="p-4">
              {recordedVideos.length === 0 ? (
                <div className="min-h-[100px] border rounded-md flex items-center justify-center">
                  <p className="text-muted-foreground text-sm">Videos will appear here after recording</p>
                </div>
              ) : (
                <div
                  ref={recordingListRef}
                  className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3 max-h-[520px] overflow-y-auto"
                >
                  {recordedVideos.map((video) => {
                    // Estimated progress (0-100) for clips still being recorded.
                    const elapsed = video.recordingStartedAt ? nowTick - video.recordingStartedAt : 0
                    const durationMs = video.recordingDurationMs || VIOLATION_CLIP_DURATION_MS
                    const progress = video.status === 'pending'
                      ? Math.min(99, Math.max(0, Math.round((elapsed / durationMs) * 100)))
                      : 100
                    const remainingSec = video.status === 'pending'
                      ? Math.max(0, Math.ceil((durationMs - elapsed) / 1000))
                      : 0

                    return (
                    <div
                      key={video.id}
                      className="border rounded-lg p-2 flex flex-col gap-2 bg-card hover:bg-accent/40 transition-colors"
                    >
                      {/* Inline video preview (fills the grid cell) */}
                      {video.status === 'ready' && video.blob ? (
                        <VideoPreview blob={video.blob} />
                      ) : (
                        <div className="w-full aspect-video rounded-md bg-muted flex items-center justify-center">
                          {video.status === 'pending' ? (
                            <span className="w-2.5 h-2.5 bg-amber-500 rounded-full animate-pulse" />
                          ) : (
                            <span className="text-xs text-muted-foreground">No preview</span>
                          )}
                        </div>
                      )}

                      {/* Badges */}
                      <div className="flex items-center gap-1 flex-wrap">
                        <Badge variant={video.type === 'exam' ? 'default' : 'destructive'}>
                          {video.type === 'exam' ? 'Exam' : 'Violation'}
                        </Badge>
                        <Badge variant="outline">{video.ext.toUpperCase()}</Badge>
                        {video.status === 'pending' && (
                          <Badge variant="outline" className="gap-1">
                            <span className="w-2 h-2 bg-amber-500 rounded-full animate-pulse" />
                            Pending
                          </Badge>
                        )}
                        {video.status === 'ready' && <Badge variant="secondary">Ready</Badge>}
                        {video.status === 'failed' && (
                          <Badge variant="outline" className="text-muted-foreground">Failed</Badge>
                        )}
                      </div>

                      {/* Filename */}
                      <p className="text-xs font-medium truncate" title={video.filename}>
                        {video.filename}
                      </p>

                      {/* Progress (pending) or meta (ready/failed) */}
                      {video.status === 'pending' ? (
                        <div>
                          <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                            <div
                              className="h-full bg-amber-500 transition-all duration-200 ease-linear"
                              style={{ width: `${progress}%` }}
                            ></div>
                          </div>
                          <div className="flex items-center justify-between mt-1 text-[11px] text-muted-foreground">
                            <span>Recording… {progress}%</span>
                            <span>~{remainingSec}s left</span>
                          </div>
                        </div>
                      ) : (
                        <div className="flex flex-col gap-0.5 text-[11px] text-muted-foreground">
                          <span>
                            {video.timestamp.toLocaleString('en-US', {
                              month: '2-digit',
                              day: '2-digit',
                              year: 'numeric',
                              hour: '2-digit',
                              minute: '2-digit',
                              second: '2-digit',
                              hour12: true
                            })}
                          </span>
                          {video.status === 'ready' && (
                            <span>{(video.size / (1024 * 1024)).toFixed(2)} MB</span>
                          )}
                        </div>
                      )}

                      {/* Download button */}
                      <Button
                        type="button"
                        size="sm"
                        variant="secondary"
                        disabled={video.status !== 'ready'}
                        onClick={(e) => {
                          e.preventDefault()
                          e.stopPropagation()
                          downloadVideo(video)
                        }}
                        className="w-full mt-auto shadow-none"
                      >
                        <Download className="w-4 h-4" />
                        {video.status === 'pending' ? 'Recording…' : 'Download'}
                      </Button>
                    </div>
                    )
                  })}
                </div>
              )}
            </CardContent>
          </div>
        )}
      </div>
    </div>
  )
}

