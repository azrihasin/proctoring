import { useRef, useState, useEffect, useCallback } from 'react'
import { FaceDetector, ObjectDetector, FilesetResolver } from '@mediapipe/tasks-vision'
import Webcam from 'react-webcam'
import { useReactMediaRecorder } from 'react-media-recorder'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { FFmpeg } from '@ffmpeg/ffmpeg'
import { fetchFile, toBlobURL } from '@ffmpeg/util'
import { fixWebmMetadata } from '@/lib/utils'
// import EKYC from '@/components/EKYC'

type DetectionType = 'cell_phone' | 'multiple_faces' | 'face_not_visible' | null

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
  
  // Exam recording state
  const [isExamActive, setIsExamActive] = useState(false)
  const [examStartTime, setExamStartTime] = useState<Date | null>(null)
  const [recordingDuration, setRecordingDuration] = useState<number>(0) // Duration in seconds
  const examVideoChunksRef = useRef<Blob[]>([])
  const examTimerIntervalRef = useRef<number | null>(null) // For timer updates
  const examSaveIntervalRef = useRef<number | null>(null) // For periodic saving
  const lastSavedSegmentTimeRef = useRef<number>(0) // Track when last segment was saved
  const [violations, setViolations] = useState<ViolationEntry[]>([])
  
  // Rolling buffer for violation recording (10 seconds before detection)
  const rollingBufferRef = useRef<Array<{ blob: Blob; timestamp: number }>>([])
  const rollingBufferRecorderRef = useRef<MediaRecorder | null>(null)
  const BUFFER_DURATION_MS = 10000 // 10 seconds
  
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
    startRecording: startExamRecording,
    stopRecording: stopExamRecording,
    mediaBlobUrl: examMediaBlobUrl,
    clearBlobUrl: clearExamBlobUrl,
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
    startRecording: startViolationRecordingHook,
    stopRecording: stopViolationRecordingHook,
    mediaBlobUrl: violationMediaBlobUrl,
    clearBlobUrl: clearViolationBlobUrl,
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
  
  const violationTimeoutRef = useRef<number | null>(null)
  const violationDetectionTimeRef = useRef<Date | null>(null)
  
  // UI state
  const [isOverlayEnabled, setIsOverlayEnabled] = useState(true)
  const [logEntries, setLogEntries] = useState<string[]>([])
  const lastTabActivityRef = useRef<number>(Date.now())
  
  // Recorded videos list
  const [recordedVideos, setRecordedVideos] = useState<RecordedVideo[]>([])

  // eKYC state
  // const [eKYCCompleted, setEKYCCompleted] = useState(false)

  // FFmpeg instance for video conversion
  const ffmpegRef = useRef<FFmpeg | null>(null)
  const [ffmpegLoaded, setFfmpegLoaded] = useState(false)

  // Get video element from webcam ref
  const getVideoElement = () => {
    return webcamRef.current?.video
  }

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

        setFfmpegLoaded(true)
        console.log('✅ FFmpeg loaded successfully')
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
    
    console.log(`✅ Video saved in original format: ${ext}`)
    return { blob: fixedWebmBlob, mime: mimeType, ext: ext as 'mp4' | 'webm', converted: false }
  }, [getBestMimeType])

  // Concatenate before+after WebM chunks (no conversion)
  const concatWebmChunks = useCallback(async (
    beforeChunks: Blob[],
    afterChunks: Blob[]
  ): Promise<SavedVideo> => {
    const mimeType = getBestMimeType()
    const beforeBlob = new Blob(beforeChunks, { type: mimeType })
    const afterBlob = new Blob(afterChunks, { type: mimeType })

    // Fix WebM metadata before concatenation
    console.log('🔧 Fixing WebM metadata for concatenation...')
    const fixedBeforeBlob = await fixWebmMetadata(beforeBlob)
    const fixedAfterBlob = await fixWebmMetadata(afterBlob)
    
    // Simply concatenate the blobs
    const concatenatedBlob = new Blob([fixedBeforeBlob, fixedAfterBlob], { type: mimeType })
    const ext = mimeType.includes('webm') ? 'webm' : 'mp4'
    
    console.log(`✅ Video chunks concatenated in original format: ${ext}`)
    return { blob: concatenatedBlob, mime: mimeType, ext: ext as 'mp4' | 'webm', converted: false }
  }, [getBestMimeType])

  // Handle webcam user media callback
  const handleUserMedia = useCallback((stream: MediaStream) => {
    mediaStreamRef.current = stream
    setWebcamReady(true)
    setWebcamError(null)
    console.log('Webcam initialized successfully')
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


  // Start rolling buffer recorder (maintains last 10 seconds)
  const startRollingBuffer = useCallback((stream: MediaStream) => {
    try {
      const mimeType = getBestMimeType()
      const options = { mimeType, videoBitsPerSecond: 1500000 }
      const recorder = new MediaRecorder(stream, options)
      rollingBufferRecorderRef.current = recorder
      rollingBufferRef.current = []

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          const now = Date.now()
          rollingBufferRef.current.push({ blob: event.data, timestamp: now })
          
          // Remove chunks older than BUFFER_DURATION_MS
          const cutoffTime = now - BUFFER_DURATION_MS
          rollingBufferRef.current = rollingBufferRef.current.filter(
            item => item.timestamp >= cutoffTime
          )
        }
      }

      recorder.onerror = (event) => {
        console.error('Rolling buffer recorder error:', event)
      }

      recorder.start(1000) // 1 second time slicing
      console.log(`✅ Rolling buffer recorder started (mimeType: ${recorder.mimeType})`)
    } catch (error) {
      console.error('Error starting rolling buffer recorder:', error)
    }
  }, [getBestMimeType])

  // Add log entry helper
  const addLogEntry = useCallback((message: string) => {
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
    setRecordedVideos(prev => [...prev, recordedVideo])
    console.log(`✅ Video added to list: ${filename} (${(savedVideo.blob.size / (1024 * 1024)).toFixed(2)} MB, type: ${savedVideo.mime}, converted: ${savedVideo.converted})`)
  }, [])

  // Process final exam video when blob becomes available from react-media-recorder
  useEffect(() => {
    if (examMediaBlobUrl && !isExamActive && examStartTime) {
      const processFinalExamVideo = async () => {
        try {
          console.log('📹 Processing final exam video from react-media-recorder...')
          // Fetch the blob from the URL
          const response = await fetch(examMediaBlobUrl!)
          const blob = await response.blob()
          
          if (blob.size === 0) {
            console.warn('⚠️ Final exam video blob is empty')
            return
          }

          console.log(`📦 Final exam video blob size: ${blob.size} bytes, type: ${blob.type}`)
          
          // Save video in original format
          addLogEntry('Saving final exam video in original format...')
          const result = await saveVideo(blob)
          
          if (result.blob.size < 50_000) {
            console.warn('⚠️ Final video too small, may be corrupted')
            addLogEntry('Final video too small, skipping save')
            return
          }

          // Generate timestamped filename
          const now = new Date()
          const year = now.getFullYear()
          const month = String(now.getMonth() + 1).padStart(2, '0')
          const day = String(now.getDate()).padStart(2, '0')
          const hours = String(now.getHours()).padStart(2, '0')
          const minutes = String(now.getMinutes()).padStart(2, '0')
          const seconds = String(now.getSeconds()).padStart(2, '0')
          
          const timestamp = `${year}${month}${day}_${hours}${minutes}${seconds}`
          const duration = Math.floor(recordingDuration / 60)
          const durationSec = recordingDuration % 60
          const filename = `exam_recording_complete_${timestamp}_${duration}m${durationSec}s.${result.ext}`
          
          // Add to recorded videos list
          addRecordedVideo(result, filename, 'exam')
          addLogEntry(`Final exam video saved: ${filename} - Added to download list`)

          console.log(`✅ Final exam video processed: ${filename}`)
          
          // Clear the blob URL
          clearExamBlobUrl()
        } catch (error) {
          console.error('❌ Error processing final exam video:', error)
          addLogEntry(`Error processing final exam video: ${error instanceof Error ? error.message : 'Unknown error'}`)
        }
      }
      
      processFinalExamVideo()
    }
  }, [examMediaBlobUrl, isExamActive, examStartTime, recordingDuration, saveVideo, addRecordedVideo, addLogEntry, clearExamBlobUrl])

  // Process violation video when blob becomes available (moved after function definitions)
  useEffect(() => {
    if (violationMediaBlobUrl && violationDetectionTimeRef.current) {
      const processViolationVideo = async () => {
        try {
          // Fetch the blob from the URL
          const response = await fetch(violationMediaBlobUrl!)
          const blob = await response.blob()
          
          console.log('🛑 Violation recorder stopped, processing video...')
          const bufferChunks = rollingBufferRef.current.map(item => item.blob)

          console.log(`📹 Buffer chunks: ${bufferChunks.length}, After blob size: ${blob.size}`)

          if (bufferChunks.length === 0 && blob.size === 0) {
            console.warn('⚠️ No video chunks available for violation recording')
            return
          }

          addLogEntry('Preparing violation video (concat)...')
          console.log('🔄 Starting concat...')

          // concat before+after chunks (no conversion)
          const result = await concatWebmChunks(bufferChunks, [blob])
          console.log(`✅ Concat complete: ${result.ext}, size: ${result.blob.size} bytes`)

          // extra safety: if video is suspiciously small, skip
          if (result.blob.size < 50_000) {
            console.warn('Video too small, skipping save')
            addLogEntry('Violation video too small, skipping save')
            return
          }

          const timestamp = violationDetectionTimeRef.current!.toISOString().replace(/[:.]/g, '-').slice(0, -5)
          const filename = `cell_phone_detection_${timestamp}.${result.ext}`

          console.log(`💾 Adding video to list: ${filename}`)
          
          // Add to recorded videos list
          addRecordedVideo(result, filename, 'violation')
          addLogEntry(`Cell phone violation video saved: ${filename} - Added to download list`)

          console.log(`✅ Violation video added to list successfully: ${filename}`)
          
          // Clear the detection time ref and blob URL
          violationDetectionTimeRef.current = null
          clearViolationBlobUrl()
        } catch (error) {
          console.error('❌ Error processing violation video:', error)
          addLogEntry(`Error processing violation video: ${error instanceof Error ? error.message : 'Unknown error'}`)
          violationDetectionTimeRef.current = null
        }
      }
      
      processViolationVideo()
    }
  }, [violationMediaBlobUrl, concatWebmChunks, addRecordedVideo, addLogEntry, getBestMimeType, clearViolationBlobUrl])

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
      setIsExamActive(true)
      const startTime = new Date()
      setExamStartTime(startTime)
      setRecordingDuration(0)
      setViolations([])
      activeFaceNotVisibleViolationRef.current = null // Reset active face_not_visible violation
      activeCellPhoneViolationRef.current = null // Reset active cell_phone violation
      activeMultipleFacesViolationRef.current = null // Reset active multiple_faces violation
      examVideoChunksRef.current = []
      lastSavedSegmentTimeRef.current = Date.now()

      // Start react-media-recorder recording
      startExamRecording()
      
      // Also create a MediaRecorder for chunk access (for periodic saving)
      const mimeType = getBestMimeType()
      const options = { mimeType, videoBitsPerSecond: 1500000 }
      const chunkRecorder = new MediaRecorder(stream, options)
      examChunkRecorderRef.current = chunkRecorder

      chunkRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          examVideoChunksRef.current.push(event.data)
        }
      }

      chunkRecorder.onerror = (event) => {
        console.error('Exam chunk recorder error:', event)
      }

      chunkRecorder.start(1000) // 1 second time slicing
      console.log(`✅ Exam recording started (mimeType: ${chunkRecorder.mimeType})`)
      
      // Start rolling buffer for violation recording
      startRollingBuffer(stream)

      // Start timer interval (update every second)
      examTimerIntervalRef.current = window.setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime.getTime()) / 1000)
        setRecordingDuration(elapsed)
      }, 1000)

      // Save video segment every 30 seconds - use inline function to avoid dependency issues
      examSaveIntervalRef.current = window.setInterval(async () => {
        if (examVideoChunksRef.current.length > 0 && isExamActive && examChunkRecorderRef.current) {
          try {
            // Request any pending data from the recorder before creating the blob
            if (examChunkRecorderRef.current.state === 'recording') {
              examChunkRecorderRef.current.requestData()
              // Wait a bit for the data to be available
              await new Promise(resolve => setTimeout(resolve, 500))
            }

            // Only save if we have enough chunks (at least 2 seconds worth)
            if (examVideoChunksRef.current.length < 2) {
              console.log('⏭️ Skipping save - not enough chunks yet')
              return
            }

            // Create a copy of chunks to save (don't clear yet)
            const chunksToSave = [...examVideoChunksRef.current]
            const recorderMimeType = examChunkRecorderRef.current?.mimeType || getBestMimeType()
            const webmBlob = new Blob(chunksToSave, { type: recorderMimeType })
            
            // Only proceed if blob has reasonable size (at least 100KB)
            if (webmBlob.size < 100_000) {
              console.log(`⏭️ Skipping save - blob too small: ${webmBlob.size} bytes`)
              return
            }

            console.log(`💾 Saving video segment: ${chunksToSave.length} chunks, ${webmBlob.size} bytes`)
            try {
              const result = await saveVideo(webmBlob)
              
              // Double-check the video has reasonable size
              if (result.blob.size < 50_000) {
                console.warn('⚠️ Video too small, skipping save')
                return
              }
              
              const now = new Date()
              const year = now.getFullYear()
              const month = String(now.getMonth() + 1).padStart(2, '0')
              const day = String(now.getDate()).padStart(2, '0')
              const hours = String(now.getHours()).padStart(2, '0')
              const minutes = String(now.getMinutes()).padStart(2, '0')
              const seconds = String(now.getSeconds()).padStart(2, '0')
              
              const timestamp = `${year}${month}${day}_${hours}${minutes}${seconds}`
              const currentDuration = Math.floor((Date.now() - startTime.getTime()) / 1000)
              const duration = Math.floor(currentDuration / 60)
              const durationSec = currentDuration % 60
              const filename = `exam_recording_${timestamp}_${duration}m${durationSec}s.${result.ext}`
              
              addRecordedVideo(result, filename, 'exam')
              addLogEntry(`Exam video segment saved: ${filename} (${duration}m ${durationSec}s)`)
              
              // Clear chunks after successful save
              examVideoChunksRef.current = []
              lastSavedSegmentTimeRef.current = Date.now()
              console.log(`✅ Video segment saved successfully: ${filename}`)
            } catch (error) {
              console.error('Error saving exam video segment:', error)
              addLogEntry(`Error saving video segment: ${error instanceof Error ? error.message : 'Unknown error'}`)
            }
          } catch (error) {
            console.error('Error in exam save interval:', error)
            addLogEntry(`Error in exam save interval: ${error instanceof Error ? error.message : 'Unknown error'}`)
          }
        }
      }, 30000) // 30 seconds
    } catch (error) {
      console.error('Error starting exam recording:', error)
      setIsExamActive(false)
    }
  }, [startRollingBuffer, saveVideo, addRecordedVideo, addLogEntry, isExamActive, startExamRecording, getBestMimeType])

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
        console.log(`📊 Blob validation - Duration: ${duration}, ReadyState: ${v.readyState}, Valid: ${isValid}`)
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
        addLogEntry('Error: Video blob is missing')
        return
      }

      if (video.blob.size === 0) {
        console.error('Video blob is empty')
        addLogEntry('Error: Video blob is empty')
        return
      }

      // Use the blob directly in its original format
      const blobToDownload = video.blob
      const filename = video.filename

      console.log('📥 Download initiated for:', filename, 'Size:', blobToDownload.size, 'Type:', blobToDownload.type)
      
      // Validate blob duration (async, but don't block download)
      validateBlobDuration(blobToDownload).then(({ duration, isValid }) => {
        if (!isValid) {
          console.warn(`⚠️ Blob may have duration/scrubbing issues (duration: ${duration})`)
          addLogEntry(`Warning: Video may not be seekable (duration: ${duration})`)
        } else {
          console.log(`✅ Blob validated successfully (duration: ${duration.toFixed(2)}s)`)
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
        console.log('🧹 Cleaned up download link and object URL')
      }, 100)

      console.log(`✅ Video download initiated: ${filename} (${(blobToDownload.size / (1024 * 1024)).toFixed(2)} MB)`)
      addLogEntry(`Video download started: ${filename}`)
    } catch (error) {
      console.error('Error downloading video:', error)
      addLogEntry(`Error downloading video: ${error instanceof Error ? error.message : 'Unknown error'}`)
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
      
      // Stop violation recorder if active (using react-media-recorder)
      if (violationRecordingStatus === 'recording') {
        stopViolationRecordingHook()
      }
      
      if (violationTimeoutRef.current) {
        clearTimeout(violationTimeoutRef.current)
        violationTimeoutRef.current = null
      }
      
      // Stop react-media-recorder exam recording
      stopExamRecording()
      
      // Stop chunk recorder and wait for final data
      const chunkRecorder = examChunkRecorderRef.current
      if (chunkRecorder && chunkRecorder.state !== 'inactive') {
        await new Promise<void>((resolve) => {
          chunkRecorder.onstop = async () => {
            await new Promise(resolve => setTimeout(resolve, 500))
            resolve()
          }
          
          if (chunkRecorder.state === 'recording') {
            chunkRecorder.requestData()
          }
          chunkRecorder.stop()
        })
      }
      
      // Save final segment if there are remaining chunks
      if (examVideoChunksRef.current.length > 0) {
        const recorderMimeType = chunkRecorder?.mimeType || getBestMimeType()
        const webmBlob = new Blob(examVideoChunksRef.current, { type: recorderMimeType })
          
          // Only save if blob has reasonable size
          if (webmBlob.size < 100_000) {
            console.warn(`⚠️ Final video blob too small: ${webmBlob.size} bytes, skipping save`)
            addLogEntry(`Final video segment too small (${webmBlob.size} bytes), skipping save`)
          } else {
            // Save final video in original format
            addLogEntry('Saving final video segment in original format...')
            try {
              const result = await saveVideo(webmBlob)
              
              // Double-check the video has reasonable size
              if (result.blob.size < 50_000) {
                console.warn('⚠️ Final video too small, skipping save')
                addLogEntry('Final video too small, skipping save')
                return
              }
              
              // Generate timestamped filename
              const now = new Date()
              const year = now.getFullYear()
              const month = String(now.getMonth() + 1).padStart(2, '0')
              const day = String(now.getDate()).padStart(2, '0')
              const hours = String(now.getHours()).padStart(2, '0')
              const minutes = String(now.getMinutes()).padStart(2, '0')
              const seconds = String(now.getSeconds()).padStart(2, '0')
              
              const timestamp = `${year}${month}${day}_${hours}${minutes}${seconds}`
              const duration = Math.floor(recordingDuration / 60)
              const durationSec = recordingDuration % 60
              const filename = `exam_recording_final_${timestamp}_${duration}m${durationSec}s.${result.ext}`
              
              // Add to recorded videos list
              addRecordedVideo(result, filename, 'exam')
              addLogEntry(`Final exam video saved: ${filename} - Added to download list`)
            } catch (error) {
              console.error('Error saving final video:', error)
              addLogEntry(`Error saving final video: ${error instanceof Error ? error.message : 'Unknown error'}`)
            }
          }
        }

        setIsExamActive(false)
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

  // Start violation recording (10 seconds after detection)
  const startViolationRecording = useCallback(async (_stream: MediaStream, detectionTime: Date) => {
    // Stop any existing violation recording
    if (violationRecordingStatus === 'recording') {
      stopViolationRecordingHook()
      // Wait a bit for it to stop
      await new Promise(resolve => setTimeout(resolve, 500))
    }

    try {
      // Store detection time for processing later
      violationDetectionTimeRef.current = detectionTime
      
      // Start react-media-recorder violation recording
      startViolationRecordingHook()
      console.log(`✅ Violation recording started (will record for 10 seconds)`)
      addLogEntry('Violation recording started (10 seconds)')
      
      // Stop recording after 10 seconds
      violationTimeoutRef.current = window.setTimeout(() => {
        console.log('⏰ Violation recording timeout reached, stopping recorder...')
        if (violationRecordingStatus === 'recording') {
          stopViolationRecordingHook()
          console.log('🛑 Violation recorder stopped')
        } else {
          console.warn('⚠️ Violation recorder is already inactive')
        }
      }, 10000)
    } catch (error) {
      console.error('Error starting violation recording:', error)
      addLogEntry(`Error starting violation recording: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }, [startViolationRecordingHook, stopViolationRecordingHook, violationRecordingStatus, addLogEntry])

  // Add violation to list when detected
  const addViolation = useCallback((type: DetectionType, score?: number) => {
    // For face_not_visible, cell_phone, and multiple_faces, track as duration - update existing or create new
    if (type === 'face_not_visible' || type === 'cell_phone' || type === 'multiple_faces') {
      setViolations(prev => {
        const activeRef = type === 'face_not_visible' 
          ? activeFaceNotVisibleViolationRef 
          : type === 'cell_phone'
          ? activeCellPhoneViolationRef
          : activeMultipleFacesViolationRef
        
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
          : 'Violation Detected'
        addLogEntry(violationMessage)
        
        console.log(`📝 ${type} violation started:`, violation)
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
      
      console.log('📝 Violation recorded:', violation)
    }
  }, [addLogEntry])

  const loadMediaPipeModels = async () => {
    try {
      console.log('Loading MediaPipe models for face and object detection...')
      
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
      console.log('✅ MediaPipe Face Detector loaded successfully')
      
      // Load Object Detector for smartphone detection
      const objectDetector = await ObjectDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
          delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        scoreThreshold: 0.3, // Lowered from 0.5 for better phone detection
        maxResults: 10 // Increased to see more detections
        // Removed categoryAllowlist to detect all objects and filter manually
      })
      
      objectDetectorRef.current = objectDetector
      console.log('✅ MediaPipe Object Detector loaded successfully')
      
      setModelsLoaded(true)

      // Start detection interval - 100ms for real-time detection (10 FPS)
      detectionIntervalRef.current = window.setInterval(() => {
        detectWithMediaPipe()
      }, 100)
      
      console.log('✅ All MediaPipe models loaded successfully')
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
                
                // ONLY draw and process cell phones - ignore all other objects
                if (!isCellPhone) return
                
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
            
            // Display cell phone detection count at bottom
            const cellPhoneCount = objectResults.detections ? 
              objectResults.detections.filter(detection => {
                if (!detection.categories || detection.categories.length === 0) return false
                const categoryLower = detection.categories[0].categoryName.toLowerCase()
                return categoryLower.includes('cell phone') || 
                       categoryLower.includes('phone') ||
                       categoryLower.includes('mobile') ||
                       categoryLower === 'cell phone'
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
          const REQUIRED_CONSECUTIVE_DETECTIONS_CELL_PHONE = 3 // 3 frames (0.3 seconds) - faster response with more accurate detection
          const REQUIRED_CONSECUTIVE_DETECTIONS_MULTIPLE_FACES = 8 // 8 frames (0.8 seconds) - reduced from 10 with better NMS
          const REQUIRED_CONSECUTIVE_FACE_MISSES = 20 // 20 frames (2.0 seconds)
          const CELL_PHONE_DEBOUNCE_MS = 1000 // 1 second debounce
          
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
                
                if (isPhone) {
                  isCellPhoneDetected = true
                  cellPhoneScore = Math.max(cellPhoneScore, category.score)
                  
                  // Always log phone detections
                  console.log(`📱 CELL PHONE DETECTED: "${category.categoryName}" (${(category.score * 100).toFixed(1)}%)`)
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
                console.log('✅ Cell phone violation ended (no phone detected)')
                return updated
              }
              return prev
            })
            activeCellPhoneViolationRef.current = null
          }

          // Check for multiple faces
          if (faceCount > 1 && !currentDetection) {
            currentDetection = 'multiple_faces'
            latestDetectionScoreRef.current = { type: 'multiple_faces', score: faceCount }
            console.log(`👥 Multiple faces detected: ${faceCount}`)
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
                console.log('✅ Multiple faces violation ended')
                return updated
              }
              return prev
            })
            activeMultipleFacesViolationRef.current = null
          }

          // Check for face not visible
          if (faceCount === 0 && !currentDetection) {
            faceNotVisibleCountRef.current++
            
            if (faceNotVisibleCountRef.current >= REQUIRED_CONSECUTIVE_FACE_MISSES) {
              currentDetection = 'face_not_visible'
              latestDetectionScoreRef.current = { type: 'face_not_visible', score: 0 }
              
              if (faceNotVisibleCountRef.current === REQUIRED_CONSECUTIVE_FACE_MISSES) {
                console.log(`⚠️ Face not visible condition met after ${faceNotVisibleCountRef.current} consecutive misses`)
              }
            } else {
              if (faceNotVisibleCountRef.current % 10 === 0) { // Log every 10 frames
                console.log(`👤 No face detected (${faceNotVisibleCountRef.current}/${REQUIRED_CONSECUTIVE_FACE_MISSES} frames)`)
              }
            }
          } else if (faceCount > 0) {
            if (faceNotVisibleCountRef.current > 0) {
              console.log(`✅ Face detected, resetting face visibility counter (was at ${faceNotVisibleCountRef.current})`)
            }
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
                    console.log('✅ Face not visible violation ended (face detected)')
                    return updated
                  }
                  return prev
                })
                activeFaceNotVisibleViolationRef.current = null
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
                  console.log('✅ Cell phone violation ended (detection type changed)')
                  return updated
                }
                return prev
              })
              activeCellPhoneViolationRef.current = null
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
                  console.log('✅ Multiple faces violation ended (detection type changed)')
                  return updated
                }
                return prev
              })
              activeMultipleFacesViolationRef.current = null
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
            
            if (!shouldTriggerAlert) {
              console.log(`⏱️ Cell phone alert debounced (${(timeSinceLastDetection / 1000).toFixed(1)}s < ${(CELL_PHONE_DEBOUNCE_MS / 1000).toFixed(0)}s)`)
            }
          }
          
          if (shouldTriggerAlert) {
            // Log violation
            if (confirmedDetection === 'cell_phone') {
              if (activeCellPhoneViolationRef.current === null) {
                console.log('⚠️ VIOLATION TRIGGERED: Cell Phone Detected!')
                
                // Start violation recording (10s before + 10s after)
                if (isExamActive && mediaStreamRef.current) {
                  const detectionTime = new Date()
                  // Fire and forget - violation recording runs in background
                  startViolationRecording(mediaStreamRef.current, detectionTime).catch((error) => {
                    console.error('Error starting violation recording:', error)
                    addLogEntry(`Error starting violation recording: ${error instanceof Error ? error.message : 'Unknown error'}`)
                  })
                }
              }
              lastCellPhoneDetectionTimeRef.current = Date.now()
            } else if (confirmedDetection === 'multiple_faces') {
              if (activeMultipleFacesViolationRef.current === null) {
                console.log('⚠️ VIOLATION TRIGGERED: Multiple Faces Detected!')
              }
            } else if (confirmedDetection === 'face_not_visible') {
              if (activeFaceNotVisibleViolationRef.current === null) {
                console.log('⚠️ VIOLATION TRIGGERED: Face Not Visible!')
              }
            }

            if (confirmedDetection) {
              const violationScore = latestDetectionScoreRef.current?.type === confirmedDetection
                ? latestDetectionScoreRef.current.score
                : undefined
              
              // Continuously update violation duration for these types
              if (confirmedDetection === 'face_not_visible' || confirmedDetection === 'cell_phone' || confirmedDetection === 'multiple_faces') {
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

  // Track tab visibility for "User Not Active On Current Tab"
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        lastTabActivityRef.current = Date.now()
      } else {
        const inactiveTime = Date.now() - lastTabActivityRef.current
        if (inactiveTime > 5000) { // 5 seconds threshold
          addLogEntry('User Not Active On Current Tab')
        }
        lastTabActivityRef.current = Date.now()
      }
    }

    const handleFocus = () => {
      lastTabActivityRef.current = Date.now()
    }

    const handleBlur = () => {
      lastTabActivityRef.current = Date.now()
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)
    window.addEventListener('focus', handleFocus)
    window.addEventListener('blur', handleBlur)

    // Check periodically for tab inactivity
    const inactivityInterval = setInterval(() => {
      if (!document.hidden && Date.now() - lastTabActivityRef.current > 5000) {
        addLogEntry('User Not Active On Current Tab')
        lastTabActivityRef.current = Date.now()
      }
    }, 10000) // Check every 10 seconds

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      window.removeEventListener('focus', handleFocus)
      window.removeEventListener('blur', handleBlur)
      clearInterval(inactivityInterval)
    }
  }, [addLogEntry])


  // Log detection history changes (for debugging/maintenance)
  useEffect(() => {
    if (detectionHistory.length > 0) {
      const latest = detectionHistory[detectionHistory.length - 1]
      console.log(`📊 Detection history: ${detectionHistory.length} entries (latest: ${latest.type} at ${latest.timestamp.toISOString()})`)
    }
  }, [detectionHistory])

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
      
      // Stop violation recorder if active (using react-media-recorder)
      if (violationRecordingStatus === 'recording') {
        stopViolationRecordingHook()
      }
      
      // Stop chunk recorder if active
      if (examChunkRecorderRef.current && examChunkRecorderRef.current.state !== 'inactive') {
        examChunkRecorderRef.current.stop()
        examChunkRecorderRef.current = null
      }
      
      if (violationTimeoutRef.current) {
        clearTimeout(violationTimeoutRef.current)
        violationTimeoutRef.current = null
      }
      
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
            Proctoring
          </h1>
        </div>

        {/* Action Bar */}
        <div className="bg-green-100 rounded-lg p-4 flex justify-center gap-4">
          <Button
            onClick={() => setIsOverlayEnabled(!isOverlayEnabled)}
            className="bg-teal-600 hover:bg-teal-700 text-white font-semibold px-6 py-2"
          >
            {isOverlayEnabled ? 'DISABLE OVERLAY' : 'ENABLE OVERLAY'}
          </Button>
        </div>

        {/* Video Feed and Log Section - Side by Side */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
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
              <h2 className="text-lg font-bold text-gray-900">Log Section</h2>
            </div>
            <CardContent className="p-4 flex-1 flex flex-col">
              <Textarea
                readOnly
                value={logEntries.join('\n')}
                className="h-full bg-white border-gray-300 font-mono text-xs"
                placeholder="Log entries will appear here..."
              />
            </CardContent>
          </Card>
        </div>

        {/* Bottom Section */}
        <div className="flex justify-end items-start gap-4">
          {/* Navigation Buttons */}
          <div className="flex gap-4 items-center">
            <Button
              className="bg-teal-600 hover:bg-teal-700 text-white font-semibold px-8 py-3"
            >
              Sebelum
            </Button>
            <Button
              onClick={isExamActive ? handleEndExam : handleStartExam}
              disabled={!webcamReady || !modelsLoaded}
              className={`font-semibold px-8 py-3 ${
                isExamActive
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              {isExamActive ? 'TAMAT PEPERIKSAAN' : 'MULA PEPERIKSAAN'}
            </Button>
          </div>
        </div>

        {/* Status Messages */}
        {!modelsLoaded && (
          <div className="text-center">
            <p className="text-blue-600 font-semibold">Loading detection models...</p>
          </div>
        )}
        {isExamActive && examStartTime && (
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
              <div className="space-y-2 max-h-[400px] overflow-y-auto">
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
      </div>
    </div>
  )
}

