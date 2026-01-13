import { useRef, useState, useEffect, useCallback } from 'react'
import { FaceDetector, ObjectDetector, FilesetResolver } from '@mediapipe/tasks-vision'
import Webcam from 'react-webcam'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'

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
  const examVideoChunksRef = useRef<Blob[]>([])
  const examMediaRecorderRef = useRef<MediaRecorder | null>(null)
  const [violations, setViolations] = useState<ViolationEntry[]>([])
  
  // UI state
  const [isOverlayEnabled, setIsOverlayEnabled] = useState(true)
  const [logEntries, setLogEntries] = useState<string[]>([])
  const [liveResults, setLiveResults] = useState<string>('No detection yet.')
  const lastTabActivityRef = useRef<number>(Date.now())

  // Get video element from webcam ref
  const getVideoElement = () => {
    return webcamRef.current?.video
  }

  // Simple MIME type - use video/webm (universally supported by MediaRecorder API)
  const RECORDING_MIME_TYPE = 'video/webm'


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
      setExamStartTime(new Date())
      setViolations([])
      activeFaceNotVisibleViolationRef.current = null // Reset active face_not_visible violation
      activeCellPhoneViolationRef.current = null // Reset active cell_phone violation
      activeMultipleFacesViolationRef.current = null // Reset active multiple_faces violation
      examVideoChunksRef.current = []

      const options = { mimeType: RECORDING_MIME_TYPE, videoBitsPerSecond: 1500000 }
      const recorder = new MediaRecorder(stream, options)
      examMediaRecorderRef.current = recorder

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          examVideoChunksRef.current.push(event.data)
        }
      }

      recorder.onerror = (event) => {
        console.error('Exam recorder error:', event)
      }

      recorder.start(1000) // 1 second time slicing
      console.log('✅ Exam recording started')
    } catch (error) {
      console.error('Error starting exam recording:', error)
      setIsExamActive(false)
    }
  }, [])

  // End exam and download video
  const handleEndExam = useCallback(() => {
    if (!isExamActive || !examMediaRecorderRef.current) {
      return
    }

    try {
      examMediaRecorderRef.current.stop()
      examMediaRecorderRef.current.onstop = () => {
        if (examVideoChunksRef.current.length > 0) {
          const blob = new Blob(examVideoChunksRef.current, { type: RECORDING_MIME_TYPE })
          
          // Generate timestamped filename
          const now = new Date()
          const year = now.getFullYear()
          const month = String(now.getMonth() + 1).padStart(2, '0')
          const day = String(now.getDate()).padStart(2, '0')
          const hours = String(now.getHours()).padStart(2, '0')
          const minutes = String(now.getMinutes()).padStart(2, '0')
          const seconds = String(now.getSeconds()).padStart(2, '0')
          
          const timestamp = `${year}${month}${day}_${hours}${minutes}${seconds}`
          const filename = `${timestamp}_exam_recording.webm`
          
          const url = URL.createObjectURL(blob)
          const a = document.createElement('a')
          document.body.appendChild(a)
          a.style.display = 'none'
          a.href = url
          a.download = filename
          a.click()
          window.URL.revokeObjectURL(url)
          document.body.removeChild(a)
          console.log(`✅ Exam video downloaded: ${filename}`)
        }

        setIsExamActive(false)
        setExamStartTime(null)
        examVideoChunksRef.current = []
        examMediaRecorderRef.current = null
      }
    } catch (error) {
      console.error('Error ending exam:', error)
      setIsExamActive(false)
    }
  }, [isExamActive])

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
            // Update the end time of the existing violation
            const updated = [...prev]
            updated[activeIndex] = {
              ...updated[activeIndex],
              endTime: new Date(), // Update end time to current time
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
            // End cell phone violation if no longer detected
            setViolations(prev => {
              const activeIndex = activeCellPhoneViolationRef.current
              if (activeIndex !== null && prev[activeIndex] && prev[activeIndex].type === 'cell_phone') {
                const updated = [...prev]
                updated[activeIndex] = {
                  ...updated[activeIndex],
                  endTime: new Date()
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
            // End multiple faces violation
            setViolations(prev => {
              const activeIndex = activeMultipleFacesViolationRef.current
              if (activeIndex !== null && prev[activeIndex] && prev[activeIndex].type === 'multiple_faces') {
                const updated = [...prev]
                updated[activeIndex] = {
                  ...updated[activeIndex],
                  endTime: new Date()
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
                    updated[activeIndex] = {
                      ...updated[activeIndex],
                      endTime: new Date()
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
            // End active violations if switching to a different type
            if (detectionCountRef.current.type === 'cell_phone' && activeCellPhoneViolationRef.current !== null) {
              setViolations(prev => {
                const activeIndex = activeCellPhoneViolationRef.current
                if (activeIndex !== null && prev[activeIndex] && prev[activeIndex].type === 'cell_phone') {
                  const updated = [...prev]
                  updated[activeIndex] = {
                    ...updated[activeIndex],
                    endTime: new Date()
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
                  updated[activeIndex] = {
                    ...updated[activeIndex],
                    endTime: new Date()
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
                updated[activeIndex] = {
                  ...updated[activeIndex],
                  endTime: new Date()
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
                updated[activeIndex] = {
                  ...updated[activeIndex],
                  endTime: new Date(),
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
                updated[activeIndex] = {
                  ...updated[activeIndex],
                  endTime: new Date()
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

  // Update live results based on detections
  useEffect(() => {
    if (violations.length > 0) {
      const latestViolation = violations[violations.length - 1]
      if (latestViolation.type === 'face_not_visible') {
        setLiveResults('Face Not Visible')
      } else if (latestViolation.type === 'cell_phone') {
        setLiveResults(`Cell Phone Detected (${((latestViolation.score || 0) * 100).toFixed(1)}%)`)
      } else if (latestViolation.type === 'multiple_faces') {
        setLiveResults(`Multiple Faces Detected (${latestViolation.score || 0} faces)`)
      }
    } else {
      setLiveResults('No detection yet.')
    }
  }, [violations])

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
      
      // Stop exam recording if active
      if (examMediaRecorderRef.current && examMediaRecorderRef.current.state !== 'inactive') {
        examMediaRecorderRef.current.stop()
        examMediaRecorderRef.current = null
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
            onClick={() => {
              if (!webcamReady) {
                // Trigger webcam initialization
                setWebcamReady(true)
              }
            }}
            disabled={webcamReady}
            className="bg-teal-600 hover:bg-teal-700 text-white font-semibold px-6 py-2"
          >
            ENABLE WEBCAM
          </Button>
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

        {/* Snapshot Section */}
        <Card className="border-purple-200">
          <div className="bg-purple-100 rounded-t-lg p-3 border-b border-purple-200">
            <h2 className="text-lg font-bold text-gray-900">Snapshot Section</h2>
          </div>
          <CardContent className="p-4">
            <div className="min-h-[150px] bg-white border border-gray-300 rounded-md flex items-center justify-center">
              <p className="text-gray-500 text-sm">No snapshots captured</p>
            </div>
          </CardContent>
        </Card>

        {/* Bottom Section */}
        <div className="flex justify-between items-start gap-4">
          {/* Live Results Panel */}
          <Card className="bg-purple-50 border-purple-200 flex-1 max-w-xs">
            <CardHeader>
              <CardTitle className="text-lg font-bold text-gray-900">Live Results</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-700">{liveResults}</p>
            </CardContent>
          </Card>

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
              className="bg-teal-600 hover:bg-teal-700 text-white font-semibold px-8 py-3"
            >
              TAMAT PEPERIKSAAN
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
          <div className="text-center">
            <p className="text-green-600 font-semibold">
              Exam in progress - Started at {examStartTime.toLocaleTimeString()}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

