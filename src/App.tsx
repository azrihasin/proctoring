import { useRef, useState, useEffect, useCallback } from 'react'
import * as cocoSsd from '@tensorflow-models/coco-ssd'
import '@tensorflow/tfjs' // Required for TensorFlow.js to work
import Webcam from 'react-webcam'
import { Button } from '@/components/ui/button'

type DetectionType = 'cell_phone' | 'prohibited_object' | 'face_not_visible' | null

type DetectionElement = { class: string; score: number; bbox: [number, number, number, number] }

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
  const faceNotVisibleCountRef = useRef<number>(0) // Track consecutive frames without person detection
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null)
  const detectionIntervalRef = useRef<number | null>(null) // Store interval ID for detection
  
  const cellPhoneDetectionDebounceRef = useRef<number | null>(null) // For 2s debounce
  const lastCellPhoneDetectionTimeRef = useRef<number>(0)
  const latestDetectionScoreRef = useRef<{ type: DetectionType; score: number } | null>(null) // Store latest detection score for violations
  const [detectionHistory, setDetectionHistory] = useState<DetectionHistoryEntry[]>([])
  const activeFaceNotVisibleViolationRef = useRef<number | null>(null) // Index of active face_not_visible violation in violations array
  const activeCellPhoneViolationRef = useRef<number | null>(null) // Index of active cell_phone violation in violations array
  
  // Exam recording state
  const [isExamActive, setIsExamActive] = useState(false)
  const [examStartTime, setExamStartTime] = useState<Date | null>(null)
  const examVideoChunksRef = useRef<Blob[]>([])
  const examMediaRecorderRef = useRef<MediaRecorder | null>(null)
  const [violations, setViolations] = useState<ViolationEntry[]>([])

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

  // Add violation to list when detected
  const addViolation = useCallback((type: DetectionType, score?: number) => {
    // For face_not_visible and cell_phone, track as duration - update existing or create new
    if (type === 'face_not_visible' || type === 'cell_phone') {
      setViolations(prev => {
        const activeRef = type === 'face_not_visible' 
          ? activeFaceNotVisibleViolationRef 
          : activeCellPhoneViolationRef
        
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
        console.log(`📝 ${type === 'face_not_visible' ? 'Face not visible' : 'Cell phone'} violation started:`, violation)
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
      console.log('📝 Violation recorded:', violation)
    }
  }, [])

  const runCoco = async () => {
    try {
      console.log('Loading COCO-SSD model optimized for fast, instant, and accurate smartphone detection at 480p...')
      
      // Using mobilenet_v2 for better accuracy while maintaining good speed
      // This model offers significantly better accuracy than lite version for cell phone detection
      // Optimized for fast, instant detection with high accuracy
      const model = await cocoSsd.load({
        base: 'mobilenet_v2', // Better accuracy for smartphone detection while maintaining good speed
      })
      
      modelRef.current = model
      console.log('✅ COCO-SSD model loaded successfully (mobilenet_v2 - optimized for fast, instant, and accurate smartphone detection)')
      
      setModelsLoaded(true)

      // Start detection interval and store the ID
      // Using 100ms interval for fast, instant detection (10 FPS detection rate)
      // Optimized for speed while maintaining accuracy
      detectionIntervalRef.current = window.setInterval(() => {
        detect()
      }, 100)
    } catch (error) {
      console.error('Error loading COCO-SSD mobilenet_v2 model:', error)
      // Fallback to lite_mobilenet_v2 if v2 fails
      try {
        console.log('Falling back to lite_mobilenet_v2...')
        const model = await cocoSsd.load({
          base: 'lite_mobilenet_v2',
        })
        modelRef.current = model
        setModelsLoaded(true)
        detectionIntervalRef.current = window.setInterval(() => {
          detect()
        }, 100)
      } catch (fallbackError) {
        console.error('Error loading fallback model:', fallbackError)
      }
    }
  }


  const detect = async () => {
    const video = getVideoElement()
    
    if (
      video &&
      video.readyState === 4 &&
      modelRef.current
    ) {
      const videoWidth = video.videoWidth
      const videoHeight = video.videoHeight

      video.width = videoWidth
      video.height = videoHeight

      if (canvasRef.current) {
        canvasRef.current.width = videoWidth
        canvasRef.current.height = videoHeight

        try {
          // COCO-SSD handles preprocessing internally, just pass the video element
          // Using mobilenet_v2 optimized for fast, instant, and accurate smartphone detection at 480p
          // Parameters: maxNumBoxes, minScore
          // Optimized minScore (35%) for faster detection while maintaining accuracy
          const predictions = await modelRef.current.detect(video, 50, 0.35) // maxNumBoxes: 50, minScore: 0.35 (35% - optimized for fast detection)
          
          // Convert COCO-SSD predictions to our format
          // COCO-SSD returns: { bbox: [x, y, width, height], class: string, score: number }
          // Our format expects: { class: string, score: number, bbox: [x, y, width, height] }
          const obj: DetectionElement[] = predictions.map((pred: cocoSsd.DetectedObject) => ({
            class: pred.class,
            score: pred.score,
            bbox: [
              pred.bbox[0],      // x
              pred.bbox[1],      // y
              pred.bbox[2],      // width
              pred.bbox[3]       // height
            ] as [number, number, number, number]
          }))

            const ctx = canvasRef.current.getContext('2d')
            if (ctx) {
              // Clear canvas
              ctx.clearRect(0, 0, videoWidth, videoHeight)

              let person_count = 0
              
              // Minimum confidence threshold for person detection
              const PERSON_CONFIDENCE_THRESHOLD = 0.60 // 60% confidence - higher accuracy with mobilenet_v2

              // First, check for person detection and draw bounding boxes
              obj.forEach((element: DetectionElement) => {
                // Only count persons with confidence above threshold
                if (element.class === 'person' && element.score >= PERSON_CONFIDENCE_THRESHOLD) {
                  person_count++
                  
                  // Bounding boxes are already scaled to video dimensions in postprocessOutput
                  const [x, y, width, height] = element.bbox
              
                  // Draw bounding box for person
                  ctx.strokeStyle = '#00ff00' // Green for person
                  ctx.lineWidth = 3
                  ctx.strokeRect(x, y, width, height)
              
                  // Draw label background
                  ctx.fillStyle = '#00ff00'
                  ctx.fillRect(x, y - 25, 150, 25)
                  
                  // Draw label text
                  ctx.fillStyle = '#ffffff'
                  ctx.font = '16px Arial'
                  ctx.fillText(
                    `Person (${(element.score * 100).toFixed(1)}%)`,
                    x + 5,
                    y - 8
                  )
                }
              })

              // Draw bounding boxes for cell phones and prohibited objects
              obj.forEach((element: DetectionElement) => {
                // Skip person as it's already drawn above
                if (element.class !== 'person') {
                  // Bounding boxes are already scaled to video dimensions in postprocessOutput
                  const [x, y, width, height] = element.bbox
                  
                  // Different colors for different object types
                  let boxColor = '#ff6b00' // Default orange
                  let labelColor = '#ff6b00'
                  
                  if (element.class === 'cell phone') {
                    boxColor = '#ff0000' // Red for cell phone (more prominent)
                    labelColor = '#ff0000'
                  } else if (element.class === 'book') {
                    boxColor = '#ff6b00' // Orange for prohibited objects
                    labelColor = '#ff6b00'
                  } else {
                    boxColor = '#ffff00' // Yellow for other objects
                    labelColor = '#ffff00'
                  }
                  
                  // Draw bounding box
                  ctx.strokeStyle = boxColor
                  ctx.lineWidth = 4 // Thicker line for better visibility
                  ctx.strokeRect(x, y, width, height)
                  
                  // Draw label background
                  ctx.fillStyle = labelColor
                  const labelWidth = Math.max(200, element.class.length * 10 + 80)
                  ctx.fillRect(x, y - 30, labelWidth, 30)
                  
                  // Draw label text
                  ctx.fillStyle = '#000000' // Black text for better contrast
                  ctx.font = 'bold 16px Arial'
                  ctx.fillText(
                    `${element.class} (${(element.score * 100).toFixed(1)}%)`,
                    x + 5,
                    y - 10
                  )
                }
              })

          // Display person count at top of canvas
          if (person_count > 0) {
            ctx.fillStyle = '#00ff00'
            ctx.fillRect(10, 10, 200, 40)
            ctx.fillStyle = '#ffffff'
            ctx.font = 'bold 18px Arial'
            ctx.fillText(
              `Person Detected`,
              15,
              35
            )
          }

          // Optimized thresholds for fast, instant, and accurate smartphone detection at 480p with 100ms intervals (10 FPS)
          const CELL_PHONE_CONFIDENCE_THRESHOLD = 0.60 // 60% confidence - optimized for fast detection while maintaining accuracy
          const PROHIBITED_OBJECT_CONFIDENCE_THRESHOLD = 0.60 // 60% confidence required
          const REQUIRED_CONSECUTIVE_DETECTIONS_CELL_PHONE = 2 // Require only 2 consecutive detections (0.2 seconds at 100ms) - instant response
          const REQUIRED_CONSECUTIVE_DETECTIONS = 3 // Require 3 consecutive detections (0.3 seconds at 100ms intervals)
          const REQUIRED_CONSECUTIVE_FACE_MISSES = 20 // Require 20 consecutive frames without person (2.0 seconds at 100ms intervals) - reliable
          const CELL_PHONE_DEBOUNCE_MS = 1000 // 1 second debounce for cell phone detections - faster repeated alerts
          
          let currentDetection: DetectionType = null
          
          // First pass: find the highest confidence cell phone detection
          let bestCellPhoneDetection: { score: number; element: DetectionElement } | null = null
          
          // Log all detections for debugging (only first few frames to avoid spam)
          const shouldLogAllDetections = Math.random() < 0.05 // Log 5% of frames
          if (shouldLogAllDetections && obj.length > 0) {
            console.log('🔍 All detections:', obj.map(d => `${d.class} (${(d.score * 100).toFixed(1)}%)`).join(', '))
          }
          
          for (const element of obj) {
            // Cell phone detection with strict validation (always check, even if face not visible)
            const className = element.class.toLowerCase().trim()
            
            // Strict matching - only accept exact 'cell phone' class name
            const isCellPhone = className === 'cell phone'
            
            if (isCellPhone) {
              // Additional validation: Check bounding box characteristics typical of cell phones
              const [, , width, height] = element.bbox
              const area = width * height
              const aspectRatio = width / height
              const videoArea = videoWidth * videoHeight
              const relativeArea = area / videoArea
              
              // Cell phones typically have:
              // - Aspect ratio between 0.35 and 3.2 (portrait or landscape, realistic phone shapes)
              // - Reasonable size (0.2% to 25% of frame for 480p detection - optimized for faster detection)
              // - Good confidence score (mobilenet_v2 provides more reliable scores)
              const isValidCellPhoneSize = relativeArea >= 0.002 && relativeArea <= 0.25 // 0.2% to 25% of frame (optimized for faster detection)
              const isValidAspectRatio = aspectRatio >= 0.35 && aspectRatio <= 3.2 // More realistic phone aspect ratios
              const hasHighConfidence = element.score >= 0.40 // Minimum 40% initial confidence (optimized for faster detection)
              
              if (isValidCellPhoneSize && isValidAspectRatio && hasHighConfidence) {
                // Log cell phone detections occasionally for debugging (reduce console spam)
                if (Math.random() < 0.2) { // Log 20% of detections
                  console.log(`📱 Cell phone detected: "${element.class}" confidence: ${(element.score * 100).toFixed(1)}% (area: ${(relativeArea * 100).toFixed(2)}%, aspect: ${aspectRatio.toFixed(2)})`)
                }
                
                // Track the best (highest confidence) cell phone detection
                if (!bestCellPhoneDetection || element.score > bestCellPhoneDetection.score) {
                  bestCellPhoneDetection = { score: element.score, element }
                }
              } else {
                // Log filtered detections occasionally for debugging
                if (element.score >= 0.60 && Math.random() < 0.1) { // Log 10% of filtered detections
                  console.log(`⚠️ Cell phone filtered: ${(element.score * 100).toFixed(1)}%, validSize=${isValidCellPhoneSize}, validAspect=${isValidAspectRatio}`)
                }
              }
            }
          }
          
          // Use the best cell phone detection if it meets the threshold
          if (bestCellPhoneDetection !== null) {
            if (bestCellPhoneDetection.score >= CELL_PHONE_CONFIDENCE_THRESHOLD) {
              // Always set currentDetection for tracking (debounce will be applied when triggering alert)
              if (!currentDetection) {
                currentDetection = 'cell_phone'
                // Store latest detection score for violation recording
                latestDetectionScoreRef.current = { type: 'cell_phone', score: bestCellPhoneDetection.score }
                console.log(`✅ Cell phone detection tracked (confidence: ${(bestCellPhoneDetection.score * 100).toFixed(1)}%, threshold: ${(CELL_PHONE_CONFIDENCE_THRESHOLD * 100).toFixed(0)}%)`)
                
                // Add to detection history
                setDetectionHistory(prev => {
                  const updated = [
                    ...prev,
                    {
                      type: 'cell_phone' as DetectionType,
                      timestamp: new Date(),
                      score: bestCellPhoneDetection.score
                    }
                  ]
                  // Keep only last 100 entries for memory management
                  return updated.slice(-100)
                })
              }
            } else if (bestCellPhoneDetection.score < CELL_PHONE_CONFIDENCE_THRESHOLD) {
              console.log(`⚠️ Cell phone detected but below threshold: ${(bestCellPhoneDetection.score * 100).toFixed(1)}% < ${(CELL_PHONE_CONFIDENCE_THRESHOLD * 100).toFixed(0)}%`)
              // End cell phone violation if it was active
              if (activeCellPhoneViolationRef.current !== null) {
                setViolations(prev => {
                  const activeIndex = activeCellPhoneViolationRef.current
                  if (activeIndex !== null && prev[activeIndex] && prev[activeIndex].type === 'cell_phone') {
                    const updated = [...prev]
                    updated[activeIndex] = {
                      ...updated[activeIndex],
                      endTime: new Date() // Final end time when cell phone no longer detected
                    }
                    console.log('✅ Cell phone violation ended (no longer detected)')
                    return updated
                  }
                  return prev
                })
                activeCellPhoneViolationRef.current = null
              }
            }
          } else {
            // No cell phone detected - end active violation if exists
            if (activeCellPhoneViolationRef.current !== null) {
              setViolations(prev => {
                const activeIndex = activeCellPhoneViolationRef.current
                if (activeIndex !== null && prev[activeIndex] && prev[activeIndex].type === 'cell_phone') {
                  const updated = [...prev]
                  updated[activeIndex] = {
                    ...updated[activeIndex],
                    endTime: new Date() // Final end time when cell phone no longer detected
                  }
                  console.log('✅ Cell phone violation ended (no longer detected)')
                  return updated
                }
                return prev
              })
              activeCellPhoneViolationRef.current = null
            }
            // Log when we're looking for phones but not finding any (occasionally)
            if (Math.random() < 0.01) { // Log 1% of frames
              console.log('🔍 Scanning for cell phones... (no detections in this frame)')
            }
          }
          
          obj.forEach((element: DetectionElement) => {

            // Prohibited object detection (books, etc.)
            // Only trigger if confidence is above threshold
            if (element.class === 'book' && element.score >= PROHIBITED_OBJECT_CONFIDENCE_THRESHOLD && !currentDetection) {
              currentDetection = 'prohibited_object'
              // Store latest detection score for violation recording
              latestDetectionScoreRef.current = { type: 'prohibited_object', score: element.score }
            }
          })

          // Check if no person detected - face not visible (only if no other violations)
          // Use a more lenient approach: require multiple consecutive misses before alerting
          if (person_count === 0 && !currentDetection) {
            // Increment the counter for consecutive frames without person
            faceNotVisibleCountRef.current++
            
            // Only set currentDetection if we've had enough consecutive misses
            // This will then go through the consecutive detection tracking below
            if (faceNotVisibleCountRef.current >= REQUIRED_CONSECUTIVE_FACE_MISSES) {
              currentDetection = 'face_not_visible'
              // Store latest detection score for violation recording (no score for face_not_visible)
              latestDetectionScoreRef.current = { type: 'face_not_visible', score: 0 }
              console.log(`⚠️ Face not visible condition met after ${faceNotVisibleCountRef.current} consecutive misses`)
            } else {
              // Log progress but don't trigger yet
              if (faceNotVisibleCountRef.current % 2 === 0) { // Log every 2 frames to reduce console spam
                console.log(`👤 No person detected (${faceNotVisibleCountRef.current}/${REQUIRED_CONSECUTIVE_FACE_MISSES} frames)`)
              }
            }
          } else if (person_count > 0) {
            // Reset counter if person is detected
            if (faceNotVisibleCountRef.current > 0) {
              const personDetection = obj.find(e => e.class === 'person')
              console.log(`✅ Person detected (confidence: ${personDetection?.score.toFixed(2) || 'N/A'}), resetting face visibility counter (was at ${faceNotVisibleCountRef.current})`)
            }
            faceNotVisibleCountRef.current = 0
            // Also reset detection count if we were tracking face_not_visible
            if (detectionCountRef.current.type === 'face_not_visible') {
              detectionCountRef.current = { type: null, count: 0 }
              // End the active face_not_visible violation by updating its end time one final time
              if (activeFaceNotVisibleViolationRef.current !== null) {
                setViolations(prev => {
                  const activeIndex = activeFaceNotVisibleViolationRef.current
                  if (activeIndex !== null && prev[activeIndex] && prev[activeIndex].type === 'face_not_visible') {
                    const updated = [...prev]
                    updated[activeIndex] = {
                      ...updated[activeIndex],
                      endTime: new Date() // Final end time when face becomes visible
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
            // Same detection type, increment count
            detectionCountRef.current.count++
          } else {
            // Different detection type or no detection, reset count
            // End active violations if switching to a different type
            if (detectionCountRef.current.type === 'cell_phone' && activeCellPhoneViolationRef.current !== null) {
              setViolations(prev => {
                const activeIndex = activeCellPhoneViolationRef.current
                if (activeIndex !== null && prev[activeIndex] && prev[activeIndex].type === 'cell_phone') {
                  const updated = [...prev]
                  updated[activeIndex] = {
                    ...updated[activeIndex],
                    endTime: new Date() // Final end time when detection type changes
                  }
                  console.log('✅ Cell phone violation ended (detection type changed)')
                  return updated
                }
                return prev
              })
              activeCellPhoneViolationRef.current = null
            }
            detectionCountRef.current.type = currentDetection
            detectionCountRef.current.count = currentDetection ? 1 : 0
          }

            // Only trigger violation after required consecutive detections
          const confirmedDetection = detectionCountRef.current.type
          // Different consecutive requirements for different violation types
          const requiredConsecutive = confirmedDetection === 'cell_phone' 
            ? REQUIRED_CONSECUTIVE_DETECTIONS_CELL_PHONE 
            : (confirmedDetection === 'face_not_visible' 
              ? REQUIRED_CONSECUTIVE_DETECTIONS 
              : REQUIRED_CONSECUTIVE_DETECTIONS)
          
          // Check if we have enough consecutive detections
          const hasEnoughConsecutiveDetections = detectionCountRef.current.count >= requiredConsecutive && !!confirmedDetection
          
          // For cell phones, also check debounce (2 seconds since last alert)
          let shouldTriggerAlert: boolean = hasEnoughConsecutiveDetections
          if (confirmedDetection === 'cell_phone' && hasEnoughConsecutiveDetections) {
            const now = Date.now()
            const timeSinceLastDetection = now - lastCellPhoneDetectionTimeRef.current
            shouldTriggerAlert = timeSinceLastDetection >= CELL_PHONE_DEBOUNCE_MS
            
            if (!shouldTriggerAlert) {
              console.log(`⏱️ Cell phone alert debounced (${(timeSinceLastDetection / 1000).toFixed(1)}s < ${(CELL_PHONE_DEBOUNCE_MS / 1000).toFixed(0)}s since last alert)`)
            }
          }
          
          if (shouldTriggerAlert) {
            const detectionTypeStr = confirmedDetection as string
            
            // Log violation
            if (detectionTypeStr === 'cell_phone') {
              // For cell_phone, we update the duration continuously
              // Only log on first trigger, subsequent updates happen silently
              if (activeCellPhoneViolationRef.current === null) {
                console.log('⚠️ VIOLATION TRIGGERED: Cell Phone Detected!')
              }
              lastCellPhoneDetectionTimeRef.current = Date.now() // Update last detection time
            } else if (detectionTypeStr === 'prohibited_object') {
              console.log('⚠️ VIOLATION TRIGGERED: Prohibited Object Detected!')
            } else if (detectionTypeStr === 'face_not_visible') {
              // For face_not_visible, we update the duration continuously
              // Only log on first trigger, subsequent updates happen silently
              if (activeFaceNotVisibleViolationRef.current === null) {
                console.log('⚠️ VIOLATION TRIGGERED: Face Not Visible!')
              }
            }

            // Always record violation to list (whether in exam mode or not)
            if (confirmedDetection) {
              const violationScore = latestDetectionScoreRef.current?.type === confirmedDetection
                ? latestDetectionScoreRef.current.score
                : undefined
              
              // For face_not_visible and cell_phone, continuously update the violation duration
              if (confirmedDetection === 'face_not_visible' || confirmedDetection === 'cell_phone') {
                addViolation(confirmedDetection, violationScore)
                // Don't reset count - keep tracking to update duration
              } else {
                addViolation(confirmedDetection, violationScore)
                // Reset detection count after recording violation (for other types)
                detectionCountRef.current.count = 0
              }
            }
          } else if (confirmedDetection === 'face_not_visible' && activeFaceNotVisibleViolationRef.current !== null) {
            // Face is still not visible, update the end time of the active violation
            setViolations(prev => {
              const activeIndex = activeFaceNotVisibleViolationRef.current
              if (activeIndex !== null && prev[activeIndex] && prev[activeIndex].type === 'face_not_visible') {
                const updated = [...prev]
                updated[activeIndex] = {
                  ...updated[activeIndex],
                  endTime: new Date() // Continuously update end time
                }
                return updated
              }
              return prev
            })
          } else if (confirmedDetection === 'cell_phone' && activeCellPhoneViolationRef.current !== null) {
            // Cell phone is still detected, update the end time of the active violation
            setViolations(prev => {
              const activeIndex = activeCellPhoneViolationRef.current
              if (activeIndex !== null && prev[activeIndex] && prev[activeIndex].type === 'cell_phone') {
                const updated = [...prev]
                updated[activeIndex] = {
                  ...updated[activeIndex],
                  endTime: new Date(), // Continuously update end time
                  score: latestDetectionScoreRef.current?.type === 'cell_phone' 
                    ? latestDetectionScoreRef.current.score 
                    : updated[activeIndex].score // Update score if available
                }
                return updated
              }
              return prev
            })
          }
            } // Close ctx if statement
          } catch (error) {
            console.error('Error during COCO-SSD inference:', error)
          }
      } // Close canvasRef.current if statement
    }
  }

  // Log detection history changes (for debugging/maintenance)
  useEffect(() => {
    if (detectionHistory.length > 0) {
      const latest = detectionHistory[detectionHistory.length - 1]
      console.log(`📊 Detection history: ${detectionHistory.length} entries (latest: ${latest.type} at ${latest.timestamp.toISOString()})`)
    }
  }, [detectionHistory])

  useEffect(() => {
    runCoco()

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

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="w-full max-w-4xl">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h1 className="text-2xl font-bold text-center mb-6 text-gray-800">
            Proctoring System
          </h1>
          
          {/* Start/End Exam Buttons */}
          <div className="flex justify-center gap-4 mb-4">
            {!isExamActive ? (
              <Button
                onClick={handleStartExam}
                disabled={!webcamReady || !modelsLoaded}
                className="bg-green-600 text-white hover:bg-green-700 font-semibold h-10 px-6"
              >
                Start Exam
              </Button>
            ) : (
              <Button
                onClick={handleEndExam}
                className="bg-red-600 text-white hover:bg-red-700 font-semibold h-10 px-6"
              >
                End Exam
              </Button>
            )}
          </div>

          {isExamActive && examStartTime && (
            <div className="text-center mb-4">
              <p className="text-green-600 font-semibold">
                Exam in progress - Started at {examStartTime.toLocaleTimeString()}
              </p>
            </div>
          )}

          <div className="flex flex-col lg:flex-row justify-center gap-4 items-start">
            <div className="relative w-full max-w-2xl aspect-video bg-black rounded-lg overflow-hidden flex-shrink-0">
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
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 w-full h-full pointer-events-none"
                style={{
                  zIndex: 10,
                }}
              />
            </div>
            
            {/* Violations JSON Display */}
            <div className="w-full lg:w-auto lg:max-w-md bg-gray-50 rounded-lg p-4 border border-gray-200 flex-shrink-0">
              <h3 className="text-lg font-semibold mb-2 text-gray-800">Violations</h3>
              {violations.length > 0 ? (
                <div className="bg-white rounded p-3 border border-gray-300 max-h-96 overflow-y-auto">
                  <pre className="text-xs text-gray-700 whitespace-pre-wrap">
                    {JSON.stringify(violations, null, 2)}
                  </pre>
                </div>
              ) : (
                <div className="bg-white rounded p-3 border border-gray-300 text-sm text-gray-500 text-center">
                  No violations recorded yet
                </div>
              )}
            </div>
          </div>
          {!modelsLoaded && (
            <p className="text-center mt-4 text-blue-600 font-semibold">
              Loading detection models...
            </p>
          )}
          {modelsLoaded && webcamReady && (
            <div className="text-center mt-4">
              {isExamActive ? (
                <p className="text-green-600 font-semibold">
                  Exam in progress - Violations are being recorded automatically
                </p>
              ) : (
                <p className="text-gray-600">
                  Detection active. Please keep your face visible and remove any prohibited items.
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

