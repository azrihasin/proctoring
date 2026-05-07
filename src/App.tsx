import { useRef, useState, useEffect, useCallback } from 'react'
import { FaceDetector, ObjectDetector, FilesetResolver } from '@mediapipe/tasks-vision'
import Webcam from 'react-webcam'
import { useReactMediaRecorder } from 'react-media-recorder'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { FFmpeg } from '@ffmpeg/ffmpeg'
import { toBlobURL } from '@ffmpeg/util'
import { fixWebmMetadata } from '@/lib/utils'
import { Eye, EyeOff, Layers, Square } from 'lucide-react'
import axios from 'axios'
// import EKYC from '@/components/EKYC'

type DetectionType = 'cell_phone' | 'multiple_faces' | 'face_not_visible' | 'tab_switch' | null

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

type RecordedVideo = {
  id: string
  filename: string
  blob: Blob
  mime: string
  ext: 'mp4' | 'webm'
  converted: boolean
  timestamp: Date
  type: 'exam' | 'violation'
  size: number
  mp4Blob?: Blob // Optional pre-converted MP4 blob
}


export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const webcamRef = useRef<Webcam>(null)
  const recordingTimeoutRef = useRef<number | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const [modelsLoaded, setModelsLoaded] = useState(false)
  const [webcamError, setWebcamError] = useState<string | null>(null)
  const [webcamReady, setWebcamReady] = useState(false)
  const detectionCountRef = useRef<{ type: DetectionType; count: number }>({ type: null, count: 0 })
  const faceNotVisibleCountRef = useRef<number>(0) // Track consecutive frames without face detection
  const faceDetectorRef = useRef<FaceDetector | null>(null)
  const objectDetectorRef = useRef<ObjectDetector | null>(null)
  const detectionIntervalRef = useRef<number | null>(null) // Store interval ID for detection
  
  const cellPhoneDetectionDebounceRef = useRef<number | null>(null) // For debounce
  const lastCellPhoneDetectionTimeRef = useRef<number>(0)
  const latestDetectionScoreRef = useRef<{ type: DetectionType; score: number } | null>(null) // Store latest detection score for violations
  const [detectionHistory, setDetectionHistory] = useState<DetectionHistoryEntry[]>([])
  const activeFaceNotVisibleViolationRef = useRef<number | null>(null) // Index of active face_not_visible violation in violations array
  const activeCellPhoneViolationRef = useRef<number | null>(null) // Index of active cell_phone violation in violations array
  const activeMultipleFacesViolationRef = useRef<number | null>(null) // Index of active multiple_faces violation in violations array
  const activeTabSwitchViolationRef = useRef<number | null>(null) // Index of active tab_switch violation in violations array
  
  // Session ID from URL parameters
  const sessionIdRef = useRef<string | null>(null)
  const [sessionIdDisplay, setSessionIdDisplay] = useState<string | null>(null)
  
  // User ID from URL parameters
  const userIdRef = useRef<string | null>(null)
  
  // Track last log time per log message type for throttling
  const lastLogTimeRef = useRef<Map<string, number>>(new Map())
  
  // Track consecutive detection counts per event type for API sending
  const eventDetectionCountRef = useRef<Map<string, number>>(new Map()) // eventType -> count
  const eventFirstDetectionRef = useRef<Map<string, boolean>>(new Map()) // eventType -> isFirstDetection
  
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
  
  // UI state
  const [isOverlayEnabled, setIsOverlayEnabled] = useState(true)
  const [logEntries, setLogEntries] = useState<string[]>([])
  const [isViewVisible, setIsViewVisible] = useState(false) // Toggle visibility for camera, log, and recorded list
  
  // Recorded videos list
  const [recordedVideos, setRecordedVideos] = useState<RecordedVideo[]>([])
  
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

  // FFmpeg instance for video conversion
  const ffmpegRef = useRef<FFmpeg | null>(null)

  // Extract URL parameters and create session ID
  const extractSessionId = useCallback(() => {
    const urlParams = new URLSearchParams(window.location.search)
    const idNo = urlParams.get('idNo')
    const userId = urlParams.get('userId') || idNo // Use userId param if available, otherwise use idNo
    
    if (idNo) {
      const sessionId = idNo
      sessionIdRef.current = sessionId
      setSessionIdDisplay(sessionId)
      userIdRef.current = userId || sessionId // Use userId if provided, otherwise use sessionId
      return sessionId
    } else {
      console.warn('⚠️ Missing URL parameter: idNo not found')
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

  // Initialize FFmpeg
  useEffect(() => {
    const loadFFmpeg = async () => {
      try {
        const ffmpeg = new FFmpeg()
        ffmpegRef.current = ffmpeg

        const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm'
        await ffmpeg.load({
          coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
          wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
        })

      } catch (error) {
        console.error('Error loading FFmpeg:', error)
      }
    }

    loadFFmpeg()
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
      
      // Convert sessionId to number (parse as integer)
      const sessionIdNum = parseInt(sessionIdRef.current, 10)
      if (isNaN(sessionIdNum)) {
        console.warn('⚠️ Session ID is not a valid number, skipping API call')
        return
      }
      
      const body = {
        sessionId: sessionIdNum,
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
    }
    return null
  }, [])

  // Reset detection count for an event type when violation ends
  const resetEventDetectionCount = useCallback((detectionType: DetectionType) => {
    const eventType = getEventTypeFromDetectionType(detectionType)
    if (eventType) {
      eventDetectionCountRef.current.set(eventType, 0)
      eventFirstDetectionRef.current.set(eventType, true) // Reset to allow first detection again
    }
  }, [getEventTypeFromDetectionType])

  // Add log entry helper with throttling for repeating logs
  const addLogEntry = useCallback((message: string) => {
    // Only add logs when exam is active
    if (!isExamActiveRef.current) {
      return
    }
    
    const now = Date.now()
    const lastLogTime = lastLogTimeRef.current.get(message) || 0
    const timeSinceLastLog = now - lastLogTime
    const THROTTLE_INTERVAL_MS = 10000 // 10 seconds
    
    // For repeating logs, only show if 10 seconds have passed since last occurrence
    if (timeSinceLastLog < THROTTLE_INTERVAL_MS && lastLogTime > 0) {
      // Skip this log entry - it's a repeat within 10 seconds
      // Don't increment API count or send API events for skipped logs
      return
    }
    
    // Update last log time for this message
    lastLogTimeRef.current.set(message, now)
    
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
    
    // Handle API sending based on consecutive log entries that actually appear in the log section
    // Only track and send API events for logs that pass the throttling check above
    const eventType = getEventTypeFromMessage(message)
    if (eventType) {
      const currentCount = eventDetectionCountRef.current.get(eventType) || 0
      const isFirstDetection = eventFirstDetectionRef.current.get(eventType) !== false
      
      // Increment count only for logs that actually appear in the log section
      const newCount = currentCount + 1
      eventDetectionCountRef.current.set(eventType, newCount)
      
      // Send to API only on first log entry or every 10th consecutive log entry of same type
      // For consecutive log entries (2nd-9th), don't send - only send 1 log total for those entries
      if (isFirstDetection) {
        // First log entry of this event type - send immediately
        eventFirstDetectionRef.current.set(eventType, false)
        sendLogToAPI(eventType, new Date()).catch((error) => {
          console.error('Error sending log to API:', error)
        })
      } else if (newCount === 10) {
        // Exactly 10th consecutive log entry - send to API and reset count to start cycle again
        eventDetectionCountRef.current.set(eventType, 0) // Reset to 0
        eventFirstDetectionRef.current.set(eventType, true) // Mark as first again for next cycle
        sendLogToAPI(eventType, new Date()).catch((error) => {
          console.error('Error sending log to API:', error)
        })
      }
      // For counts 2-9, don't send anything - only 1 log was sent on first log entry
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

  // Add recorded video to list
  const addRecordedVideo = useCallback((savedVideo: SavedVideo, filename: string, type: 'exam' | 'violation') => {
    // Store the blob directly - React state will keep it in memory
    // The blob should remain valid as long as it's in state
    const recordedVideo: RecordedVideo = {
      id: `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      filename,
      blob: savedVideo.blob, // Store blob directly
      mime: savedVideo.mime,
      ext: savedVideo.ext,
      converted: savedVideo.converted,
      timestamp: new Date(),
      type,
      size: savedVideo.blob.size,
      mp4Blob: savedVideo.converted && savedVideo.ext === 'mp4' ? savedVideo.blob : undefined,
    }
    
    setRecordedVideos(prev => {
      const newList = [...prev, recordedVideo]
      return newList
    })
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

      // Reset detection counts and first detection flags when exam starts
      eventDetectionCountRef.current.clear()
      eventFirstDetectionRef.current.clear()

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

  // Simple synchronous download helper - must be called directly from user click
  const downloadVideo = useCallback((video: RecordedVideo) => {
    try {
      if (!video.blob) {
        console.error('Video blob is null or undefined')
        return
      }

      if (video.blob.size === 0) {
        console.error('Video blob is empty')
        return
      }

      // Use the blob directly in its original format
      const blobToDownload = video.blob
      const filename = video.filename

      // Validate blob duration (async, but don't block download)
      validateBlobDuration(blobToDownload).then(({ duration, isValid }) => {
        if (!isValid) {
          console.warn(`⚠️ Blob may have duration/scrubbing issues (duration: ${duration})`)
        }
      }).catch(err => {
        console.warn('Blob validation error:', err)
      })

      // Create download link synchronously
      const url = URL.createObjectURL(blobToDownload)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
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
  }, [addLogEntry, validateBlobDuration])

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
      
      // Reset detection counts when exam ends
      eventDetectionCountRef.current.clear()
      eventFirstDetectionRef.current.clear()
      
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
    
    // Mark this violation type as being recorded
    activeRecordingByTypeRef.current.set(violationType, true)
    
    // Initialize chunks array for this violation type
    violationChunksByTypeRef.current.set(violationType, [])
    violationDetectionTimesRef.current.set(violationType, detectionTime)

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
        
        const chunks = violationChunksByTypeRef.current.get(currentViolationType) || []
        const detectionTime = violationDetectionTimesRef.current.get(currentViolationType)
        
        // Process video directly here to avoid closure issues
        if (chunks.length === 0) {
          console.error(`❌❌❌ NO CHUNKS TO PROCESS for ${currentViolationType} - RECORDING FAILED`)
          activeRecordingByTypeRef.current.set(currentViolationType, false)
          violationRecordersRef.current.delete(currentViolationType)
          violationChunksByTypeRef.current.delete(currentViolationType)
          violationDetectionTimesRef.current.delete(currentViolationType)
          return
        }
        
        if (!detectionTime) {
          console.error(`❌ Missing detection time for ${currentViolationType}`)
          activeRecordingByTypeRef.current.set(currentViolationType, false)
          violationRecordersRef.current.delete(currentViolationType)
          violationChunksByTypeRef.current.delete(currentViolationType)
          violationDetectionTimesRef.current.delete(currentViolationType)
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
            return
          }
          
          // Fix metadata
          const fixedBlob = await fixWebmMetadata(blob)
          
          const ext = currentMimeType.includes('webm') ? 'webm' : 'mp4'
          const result: SavedVideo = { 
            blob: fixedBlob, 
            mime: currentMimeType, 
            ext: ext as 'mp4' | 'webm', 
            converted: false 
          }
          
          const timestamp = detectionTime.toISOString().replace(/[:.]/g, '-').slice(0, -5)
          const violationTypeStr = currentViolationType === 'cell_phone' ? 'cell_phone_detection' :
                                   currentViolationType === 'multiple_faces' ? 'multiple_faces_detection' :
                                   currentViolationType === 'face_not_visible' ? 'face_not_visible_detection' :
                                   currentViolationType === 'tab_switch' ? 'tab_switch_detection' :
                                   'violation'
          const filename = `${violationTypeStr}_${timestamp}.${result.ext}`

          // Add to recorded videos list immediately
          addRecordedVideo(result, filename, 'violation')
          
          // Map violation type to eventType format
          const eventTypeMap: Record<string, string> = {
            'cell_phone': 'phone',
            'face_not_visible': 'face-not-visible',
            'multiple_faces': 'multiple-faces',
            'tab_switch': 'tab-switch'
          }
          const eventType = eventTypeMap[currentViolationType] || currentViolationType || 'unknown'
          
          // Calculate startTime and endTime for the recording
          // Recording starts at detectionTime and stops after 10 seconds
          const startTime = detectionTime
          const endTime = new Date(detectionTime.getTime() + 10000) // 10 seconds after detection
          
          // Send video to API
          sendVideoToAPI(fixedBlob, eventType, startTime, endTime).catch((error) => {
            console.error('Error sending video to API:', error)
          })
          
          // Clear refs and recording flag for this violation type
          activeRecordingByTypeRef.current.set(currentViolationType, false)
          violationRecordersRef.current.delete(currentViolationType)
          violationChunksByTypeRef.current.delete(currentViolationType)
          violationDetectionTimesRef.current.delete(currentViolationType)
          violationTimeoutsRef.current.delete(currentViolationType)
        } catch (error) {
          console.error(`❌❌❌ ERROR in onstop processing for ${currentViolationType}:`, error)
          // Clear recording flag even on error
          activeRecordingByTypeRef.current.set(currentViolationType, false)
          violationRecordersRef.current.delete(currentViolationType)
          violationChunksByTypeRef.current.delete(currentViolationType)
          violationDetectionTimesRef.current.delete(currentViolationType)
          violationTimeoutsRef.current.delete(currentViolationType)
        }
      }
      
      recorder.onerror = (event) => {
        console.error(`❌ Violation recorder error for ${violationType}:`, event)
        activeRecordingByTypeRef.current.set(violationType, false)
        violationRecordersRef.current.delete(violationType)
        violationChunksByTypeRef.current.delete(violationType)
        violationDetectionTimesRef.current.delete(violationType)
        violationTimeoutsRef.current.delete(violationType)
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
    }
  }, [addLogEntry, getBestMimeType, addRecordedVideo, validateBlobDuration, sendVideoToAPI])


  // Add violation to list when detected
  const addViolation = useCallback((type: DetectionType, score?: number) => {
    // For face_not_visible, cell_phone, multiple_faces, and tab_switch, track as duration - update existing or create new
    if (type === 'face_not_visible' || type === 'cell_phone' || type === 'multiple_faces' || type === 'tab_switch') {
      setViolations(prev => {
        const activeRef = type === 'face_not_visible' 
          ? activeFaceNotVisibleViolationRef 
          : type === 'cell_phone'
          ? activeCellPhoneViolationRef
          : type === 'multiple_faces'
          ? activeMultipleFacesViolationRef
          : activeTabSwitchViolationRef
        
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
          ? `Cell Phone Detected (${((score || 0) * 100).toFixed(1)}%)`
          : type === 'multiple_faces'
          ? 'Multiple Faces Detected'
          : type === 'tab_switch'
          ? 'Tab Switch Detected'
          : 'Violation Detected'
        addLogEntry(violationMessage)
        
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
  }, [addLogEntry])

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

          const ctx = canvasRef.current.getContext('2d')
          if (ctx) {
            // Clear canvas
            ctx.clearRect(0, 0, videoWidth, videoHeight)

            const faceCount = faceResults.detections.length
            
            // Draw face detections
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

            // Draw object detections - ONLY cell phones
            if (objectResults.detections && objectResults.detections.length > 0) {
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

            // Display detection status at top of canvas
            if (faceCount > 1) {
              ctx.fillStyle = '#ff0000'
              ctx.fillRect(10, 10, 220, 40)
              ctx.fillStyle = '#ffffff'
              ctx.font = 'bold 18px Arial'
              ctx.fillText(
                `Multiple Faces: ${faceCount}`,
                15,
                35
              )
            } else if (faceCount === 1) {
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
            
            // Display cell phone detection count at bottom (only count detections with >=42% confidence)
            const cellPhoneCount = objectResults.detections ? 
              objectResults.detections.filter(detection => {
                if (!detection.categories || detection.categories.length === 0) return false
                const category = detection.categories[0]
                const categoryLower = category.categoryName.toLowerCase()
                const isPhone = categoryLower.includes('cell phone') || 
                               categoryLower.includes('phone') ||
                               categoryLower.includes('mobile') ||
                               categoryLower === 'cell phone'
                return isPhone && category.score >= 0.42
              }).length : 0
            
            if (cellPhoneCount > 0) {
              ctx.fillStyle = 'rgba(255, 0, 255, 0.7)'
              ctx.fillRect(10, videoHeight - 50, 250, 40)
              ctx.fillStyle = '#ffffff'
              ctx.font = 'bold 14px Arial'
              ctx.fillText(
                `Cell phones detected: ${cellPhoneCount}`,
                15,
                videoHeight - 25
              )
            }

          // Detection thresholds for MediaPipe
          const REQUIRED_CONSECUTIVE_DETECTIONS_CELL_PHONE = 2 // 2 frames (0.2 seconds) - smooth detection with higher confidence
          const REQUIRED_CONSECUTIVE_DETECTIONS_MULTIPLE_FACES = 8 // 8 frames (0.8 seconds) - reduced from 10 with better NMS
          const REQUIRED_CONSECUTIVE_FACE_MISSES = 20 // 20 frames (2.0 seconds)
          const CELL_PHONE_DEBOUNCE_MS = 800 // 0.8 seconds debounce for smoother detection
          
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
            resetEventDetectionCount('cell_phone') // Reset detection count when violation ends
          }

          // Check for multiple faces
          if (faceCount > 1 && !currentDetection) {
            currentDetection = 'multiple_faces'
            latestDetectionScoreRef.current = { type: 'multiple_faces', score: faceCount }
          } else if (faceCount <= 1 && activeMultipleFacesViolationRef.current !== null) {
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
                
                // Violation recording stops automatically after 10 seconds from detection
                // No need to schedule stop here
                
                return updated
              }
              return prev
            })
            activeMultipleFacesViolationRef.current = null
            resetEventDetectionCount('multiple_faces') // Reset detection count when violation ends
          }

          // Check for face not visible
          if (faceCount === 0 && !currentDetection) {
            faceNotVisibleCountRef.current++
            
            if (faceNotVisibleCountRef.current >= REQUIRED_CONSECUTIVE_FACE_MISSES) {
              currentDetection = 'face_not_visible'
              latestDetectionScoreRef.current = { type: 'face_not_visible', score: 0 }
              
            }
          } else if (faceCount > 0) {
            faceNotVisibleCountRef.current = 0
            
            if (detectionCountRef.current.type === 'face_not_visible') {
              detectionCountRef.current = { type: null, count: 0 }
              
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
                    
                    // Violation recording stops automatically after 10 seconds from detection
                    // No need to schedule stop here
                    
                    return updated
                  }
                  return prev
                })
                activeFaceNotVisibleViolationRef.current = null
                resetEventDetectionCount('face_not_visible') // Reset detection count when violation ends
              }
            }
          }

          // Track consecutive detections to reduce false positives
          if (currentDetection === detectionCountRef.current.type) {
            detectionCountRef.current.count++
          } else {
            // End active violations if switching to a different type (add 10 seconds after)
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
              resetEventDetectionCount('cell_phone') // Reset detection count when violation ends
            }
            
            if (detectionCountRef.current.type === 'multiple_faces' && activeMultipleFacesViolationRef.current !== null) {
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
              resetEventDetectionCount('multiple_faces') // Reset detection count when violation ends
            }
            
            detectionCountRef.current.type = currentDetection
            detectionCountRef.current.count = currentDetection ? 1 : 0
          }

          // Only trigger violation after required consecutive detections
          const confirmedDetection = detectionCountRef.current.type
          const requiredConsecutive = confirmedDetection === 'cell_phone' 
            ? REQUIRED_CONSECUTIVE_DETECTIONS_CELL_PHONE 
            : confirmedDetection === 'multiple_faces'
            ? REQUIRED_CONSECUTIVE_DETECTIONS_MULTIPLE_FACES
            : REQUIRED_CONSECUTIVE_FACE_MISSES
          
          const hasEnoughConsecutiveDetections = detectionCountRef.current.count >= requiredConsecutive && !!confirmedDetection
          
          // For cell phones, also check debounce
          let shouldTriggerAlert: boolean = hasEnoughConsecutiveDetections
          if (confirmedDetection === 'cell_phone' && hasEnoughConsecutiveDetections) {
            const now = Date.now()
            const timeSinceLastDetection = now - lastCellPhoneDetectionTimeRef.current
            shouldTriggerAlert = timeSinceLastDetection >= CELL_PHONE_DEBOUNCE_MS
            
          }
          
          if (shouldTriggerAlert) {
            // Log violation
            if (confirmedDetection === 'cell_phone') {
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
            } else if (confirmedDetection === 'multiple_faces') {
              // Start violation recording automatically (will be ignored if already recording this type)
              const streamToUse = getMediaStream()
              const examIsActive = isExamActiveRef.current || isExamActive // Check both ref and state
              
              if (examIsActive && streamToUse) {
                const detectionTime = new Date()
                // startViolationRecording will ignore if same type is already recording
                startViolationRecording(streamToUse, detectionTime, 'multiple_faces').catch((error) => {
                  console.error('Error starting violation recording:', error)
                })
              }
            } else if (confirmedDetection === 'face_not_visible') {
              // Start violation recording automatically (will be ignored if already recording this type)
              const streamToUse = getMediaStream()
              const examIsActive = isExamActiveRef.current || isExamActive // Check both ref and state
              
              if (examIsActive && streamToUse) {
                const detectionTime = new Date()
                // startViolationRecording will ignore if same type is already recording
                startViolationRecording(streamToUse, detectionTime, 'face_not_visible').catch((error) => {
                  console.error('Error starting violation recording:', error)
                })
              }
            }

            if (confirmedDetection) {
              const violationScore = latestDetectionScoreRef.current?.type === confirmedDetection
                ? latestDetectionScoreRef.current.score
                : undefined
              
              // Continuously update violation duration for these types
              if (confirmedDetection === 'face_not_visible' || confirmedDetection === 'cell_phone' || confirmedDetection === 'multiple_faces' || confirmedDetection === 'tab_switch') {
                addViolation(confirmedDetection, violationScore)
              } else {
                addViolation(confirmedDetection, violationScore)
                detectionCountRef.current.count = 0
              }
            }
          } else if (confirmedDetection === 'face_not_visible' && activeFaceNotVisibleViolationRef.current !== null) {
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
          } else if (confirmedDetection === 'multiple_faces' && activeMultipleFacesViolationRef.current !== null) {
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
          }
          } // Close ctx if statement
        } catch (error) {
          console.error('Error during MediaPipe detection:', error)
        }
      } // Close canvasRef.current if statement
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
        resetEventDetectionCount('tab_switch') // Reset detection count when violation ends
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
    extractSessionId()
  }, [extractSessionId])

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
      
      // Clear detection counts on unmount
      eventDetectionCountRef.current.clear()
      eventFirstDetectionRef.current.clear()
      
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
    <div className="min-h-screen bg-slate-50 p-4">
      <div className="max-w-7xl mx-auto space-y-4">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900">
            {' '}
          </h1>
        </div>

        {/* Control Buttons - Outside Cards */}
        <div className="flex items-center justify-center gap-3">
          <Button
            onClick={isExamActive ? handleEndExam : handleStartExam}
            disabled={!webcamReady || !modelsLoaded}
            style={{ display: 'none' }}
            className={`font-semibold px-8 py-3 ${
              isExamActive
                ? 'bg-red-600 hover:bg-red-700 text-white'
                : 'bg-green-600 hover:bg-green-700 text-white'
            }`}
          >
            {isExamActive ? 'TAMAT PEPERIKSAAN' : 'MULA PEPERIKSAAN'}
          </Button>
          <Button
            onClick={() => setIsViewVisible(!isViewVisible)}
            variant="ghost"
            size="icon"
            className="bg-transparent hover:bg-transparent text-current hover:text-current border-none shadow-none cursor-pointer"
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
          <Card className="bg-orange-50 border-orange-200 lg:col-span-2">
            <CardContent className="p-0">
              <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
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
                {isOverlayEnabled && (
                  <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 w-full h-full pointer-events-none"
                    style={{ zIndex: 10 }}
                  />
                )}
              </div>
            </CardContent>
          </Card>

          {/* Log Section - Takes 1 column */}
          <Card className="border-green-200 flex flex-col">
            <div className="bg-green-100 rounded-t-lg p-3 border-b border-green-200">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-3">
                  <h2 className="text-lg font-bold text-gray-900">Log Section</h2>
                  <div className="text-xs text-gray-600 bg-gray-100 px-3 py-1.5 rounded-md border border-gray-300">
                    <span className="font-semibold text-gray-700">Session ID:</span> {sessionIdDisplay || 'No Session ID'}
                  </div>
                </div>
                <Button
                  onClick={() => setIsOverlayEnabled(!isOverlayEnabled)}
                  size="icon"
                  className="bg-teal-600 hover:bg-teal-700 text-white"
                  title={isOverlayEnabled ? 'Disable Overlay' : 'Enable Overlay'}
                >
                  {isOverlayEnabled ? (
                    <Layers className="w-5 h-5" />
                  ) : (
                    <Square className="w-5 h-5" />
                  )}
                </Button>
              </div>
            </div>
            <CardContent className="p-4 flex-1 flex flex-col">
              <Textarea
                ref={logSectionRef}
                readOnly
                value={logEntries.join('\n')}
                className="h-full bg-white border-gray-300 font-mono text-xs overflow-y-auto"
                placeholder="Log entries will appear here..."
              />
            </CardContent>
          </Card>
        </div>

        {/* Status Messages */}
        {!modelsLoaded && (
          <div className="text-center">
            <p className="text-blue-600 font-semibold">Loading detection models...</p>
          </div>
        )}

        {/* Recording Timer - Inside show/hide section */}
        {isViewVisible && isExamActive && examStartTime && (
          <div className="text-center bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="flex items-center justify-center gap-3">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                <p className="text-green-700 font-semibold text-lg">Recording in progress</p>
              </div>
              <div className="text-green-600 font-mono text-xl font-bold">
                {Math.floor(recordingDuration / 3600).toString().padStart(2, '0')}:
                {Math.floor((recordingDuration % 3600) / 60).toString().padStart(2, '0')}:
                {(recordingDuration % 60).toString().padStart(2, '0')}
              </div>
            </div>
            <p className="text-sm text-green-600 mt-1">
              Started at {examStartTime.toLocaleTimeString()}
            </p>
          </div>
        )}

        {/* Recorded Videos List */}
        {isViewVisible && (
          <Card className="border-blue-200">
            <div className="bg-blue-100 rounded-t-lg p-3 border-b border-blue-200">
              <h2 className="text-lg font-bold text-gray-900">Recorded Videos</h2>
              <p className="text-sm text-gray-600 mt-1">
                {recordedVideos.length > 0 
                  ? `${recordedVideos.length} video${recordedVideos.length > 1 ? 's' : ''} recorded`
                  : 'No videos recorded yet'}
              </p>
            </div>
            <CardContent className="p-4">
              {recordedVideos.length === 0 ? (
                <div className="min-h-[100px] bg-white border border-gray-300 rounded-md flex items-center justify-center">
                  <p className="text-gray-500 text-sm">Videos will appear here after recording</p>
                </div>
              ) : (
                <div ref={recordingListRef} className="space-y-2 max-h-[400px] overflow-y-auto">
                  {recordedVideos.map((video) => (
                    <div
                      key={video.id}
                      className="bg-white border border-gray-300 rounded-md p-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span
                            className={`px-2 py-1 rounded text-xs font-semibold ${
                              video.type === 'exam'
                                ? 'bg-green-100 text-green-800'
                                : 'bg-red-100 text-red-800'
                            }`}
                          >
                            {video.type === 'exam' ? 'Exam' : 'Violation'}
                          </span>
                          <span
                            className={`px-2 py-1 rounded text-xs font-semibold ${
                              video.converted
                                ? 'bg-blue-100 text-blue-800'
                                : 'bg-yellow-100 text-yellow-800'
                            }`}
                          >
                            {video.ext.toUpperCase()}
                          </span>
                        </div>
                        <p className="text-sm font-medium text-gray-900 truncate">{video.filename}</p>
                        <div className="flex items-center gap-4 mt-1 text-xs text-gray-500">
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
                          <span>
                            {(video.size / (1024 * 1024)).toFixed(2)} MB
                          </span>
                        </div>
                      </div>
                      <div className="ml-4 flex gap-2">
                        <Button
                          type="button"
                          onClick={(e) => {
                            e.preventDefault()
                            e.stopPropagation()
                            downloadVideo(video)
                          }}
                          className="bg-teal-600 hover:bg-teal-700 text-white font-semibold px-4 py-2 text-sm"
                        >
                          Download
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

