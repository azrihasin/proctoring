import { useRef, useState, useEffect } from 'react'
import * as tf from '@tensorflow/tfjs'
import * as cocossd from '@tensorflow-models/coco-ssd'
import Webcam from 'react-webcam'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'

type DetectionType = 'cell_phone' | 'prohibited_object' | 'face_not_visible' | null

export default function App() {
  const webcamRef = useRef<Webcam>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const recordedChunksRef = useRef<Blob[]>([])
  const recordingTimeoutRef = useRef<number | null>(null)
  const [isDialogOpen, setIsDialogOpen] = useState(false)
  const [detectionMessage, setDetectionMessage] = useState('')
  const [detectionTitle, setDetectionTitle] = useState('')
  const [modelsLoaded, setModelsLoaded] = useState(false)
  const [webcamError, setWebcamError] = useState<string | null>(null)
  const [webcamReady, setWebcamReady] = useState(false)
  const [recordedVideoBlob, setRecordedVideoBlob] = useState<Blob | null>(null)
  const [isRecording, setIsRecording] = useState(false)

  const startRecording = () => {
    if (!webcamRef.current?.video) return

    const videoStream = webcamRef.current.video.srcObject as MediaStream
    if (!videoStream) return

    try {
      // Clear previous recording chunks and blob
      recordedChunksRef.current = []
      setRecordedVideoBlob(null)

      // Create MediaRecorder with video/webm codec
      const mediaRecorder = new MediaRecorder(videoStream, {
        mimeType: 'video/webm;codecs=vp9',
      })

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          recordedChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' })
        setRecordedVideoBlob(blob)
        setIsRecording(false)
        console.log('Recording stopped, video blob created')
      }

      mediaRecorderRef.current = mediaRecorder
      mediaRecorder.start()
      setIsRecording(true)
      console.log('Recording started')

      // Stop recording after 5 seconds
      recordingTimeoutRef.current = window.setTimeout(() => {
        stopRecording()
      }, 5000)
    } catch (error) {
      console.error('Error starting recording:', error)
      // Fallback to different codec if vp9 is not supported
      try {
        const mediaRecorder = new MediaRecorder(videoStream, {
          mimeType: 'video/webm',
        })
        recordedChunksRef.current = []
        setRecordedVideoBlob(null)
        mediaRecorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) {
            recordedChunksRef.current.push(event.data)
          }
        }
        mediaRecorder.onstop = () => {
          const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' })
          setRecordedVideoBlob(blob)
          setIsRecording(false)
        }
        mediaRecorderRef.current = mediaRecorder
        mediaRecorder.start()
        setIsRecording(true)
        recordingTimeoutRef.current = window.setTimeout(() => {
          stopRecording()
        }, 5000)
      } catch (fallbackError) {
        console.error('Error with fallback recording:', fallbackError)
      }
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
    if (recordingTimeoutRef.current) {
      clearTimeout(recordingTimeoutRef.current)
      recordingTimeoutRef.current = null
    }
  }

  const downloadRecording = () => {
    if (!recordedVideoBlob) return

    const url = URL.createObjectURL(recordedVideoBlob)
    const a = document.createElement('a')
    a.href = url
    a.download = `violation-recording-${new Date().toISOString().replace(/[:.]/g, '-')}.webm`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const runCoco = async () => {
    try {
      // Initialize TensorFlow.js backend
      await tf.ready()
      const backend = tf.getBackend()
      console.log('TensorFlow.js backend initialized:', backend)
      
      if (backend !== 'webgl') {
        console.warn('WebGL not available, using CPU backend (slower). Consider enabling WebGL for better performance.')
      } else {
        console.log('GPU acceleration enabled via WebGL')
      }

      // Load COCO-SSD model with lite_mobilenet_v2 for faster inference
      // This base model is optimized for speed while maintaining good accuracy for cell phones
      const net = await cocossd.load({
        base: 'lite_mobilenet_v2' // Much faster than default mobilenet_v2
      })
      console.log('AI models loaded.')
      setModelsLoaded(true)

      setInterval(() => {
        detect(net)
      }, 500) // Reduced to 500ms for more frequent detection (better for small objects like phones)
    } catch (error) {
      console.error('Error loading models:', error)
    }
  }

  const detect = async (net: cocossd.ObjectDetection) => {
    if (
      typeof webcamRef.current !== 'undefined' &&
      webcamRef.current !== null &&
      webcamRef.current.video &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video
      const videoWidth = webcamRef.current.video.videoWidth
      const videoHeight = webcamRef.current.video.videoHeight

      webcamRef.current.video.width = videoWidth
      webcamRef.current.video.height = videoHeight

      if (canvasRef.current) {
        canvasRef.current.width = videoWidth
        canvasRef.current.height = videoHeight

        const obj = await net.detect(video)

        const ctx = canvasRef.current.getContext('2d')
        if (ctx) {
          // Clear canvas
          ctx.clearRect(0, 0, videoWidth, videoHeight)

          let person_count = 0
          let detectedType: DetectionType = null
          let title = ''
          let message = ''

          // First, check for person detection and draw bounding boxes
          obj.forEach((element) => {
            if (element.class === 'person') {
              person_count++
              
              // Draw bounding box for person
              const [x, y, width, height] = element.bbox
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
          obj.forEach((element) => {
            // Skip person as it's already drawn above
            if (element.class !== 'person') {
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

          // Check for cell phones and prohibited objects first (regardless of person visibility)
          obj.forEach((element) => {
            // Cell phone detection (always check, even if face not visible)
            if (element.class === 'cell phone' && !detectedType) {
              detectedType = 'cell_phone'
              title = 'Cell Phone Detected'
              message = 'Action has been Recorded'
            }

            // Prohibited object detection (books, etc.)
            if (element.class === 'book' && !detectedType) {
              detectedType = 'prohibited_object'
              title = 'Prohibited Object Detected'
              message = 'Action has been Recorded'
            }
          })

          // Check if no person detected - face not visible (only if no other violations)
          if (person_count === 0 && !detectedType) {
            detectedType = 'face_not_visible'
            title = 'Face Not Visible'
            message = 'Action has been Recorded'
          }

          // Show dialog if detection found
          if (detectedType) {
            setDetectionTitle(title)
            setDetectionMessage(message)
            setIsDialogOpen(true)
            // Start recording when violation is detected
            if (!isRecording) {
              startRecording()
            }
          }
        }
      }
    }
  }

  useEffect(() => {
    runCoco()

    // Cleanup on unmount
    return () => {
      if (recordingTimeoutRef.current) {
        clearTimeout(recordingTimeoutRef.current)
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop()
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
          <div className="flex justify-center">
            <div className="relative w-full max-w-2xl aspect-video bg-black rounded-lg overflow-hidden">
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
                    muted={true}
                    screenshotFormat="image/jpeg"
                    videoConstraints={{
                      width: { ideal: 640 }, // Reduced resolution for faster processing (still good quality)
                      height: { ideal: 480 },
                      facingMode: 'user',
                    }}
                    onUserMedia={() => {
                      setWebcamReady(true)
                      setWebcamError(null)
                      console.log('Webcam connected successfully')
                    }}
                    onUserMediaError={(error) => {
                      console.error('Webcam error:', error)
                      const errorName = error instanceof DOMException ? error.name : typeof error === 'string' ? error : 'UnknownError'
                      setWebcamError(
                        errorName === 'NotAllowedError'
                          ? 'Camera access denied. Please allow camera access in your browser settings.'
                          : errorName === 'NotFoundError'
                          ? 'No camera found. Please connect a camera and try again.'
                          : 'Failed to access camera. Please check your camera permissions.'
                      )
                      setWebcamReady(false)
                    }}
                    className="w-full h-full object-cover"
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
          </div>
          {!modelsLoaded && (
            <p className="text-center mt-4 text-blue-600 font-semibold">
              Loading detection models...
            </p>
          )}
          {modelsLoaded && webcamReady && (
            <div className="text-center mt-4">
              <p className="text-gray-600">
                Detection active. Please keep your face visible and remove any prohibited items.
              </p>
              {isRecording && (
                <p className="text-red-600 font-semibold mt-2 flex items-center justify-center gap-2">
                  <span className="inline-block w-3 h-3 bg-red-600 rounded-full animate-pulse"></span>
                  Recording violation...
                </p>
              )}
            </div>
          )}
        </div>
      </div>

      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="sm:max-w-[425px] bg-white border-gray-200 [&>button]:hidden p-6">
          <DialogHeader className="text-left space-y-0 pb-0">
            <DialogTitle className="text-left font-bold text-lg text-gray-900 mb-3 leading-tight">
              {detectionTitle}
            </DialogTitle>
            <DialogDescription className="text-left text-sm text-gray-600 leading-relaxed mt-0">
              {detectionMessage}
              {isRecording && (
                <span className="block mt-2 text-red-600 font-semibold">
                  Recording in progress... (5 seconds)
                </span>
              )}
              {recordedVideoBlob && !isRecording && (
                <span className="block mt-2 text-green-600 font-semibold">
                  Recording complete! Ready to download.
                </span>
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="sm:justify-end mt-6 pt-0 gap-2">
            {recordedVideoBlob && !isRecording && (
              <Button
                variant="outline"
                className="border-green-600 text-green-700 bg-white hover:bg-green-50 font-medium h-9 px-4"
                onClick={downloadRecording}
              >
                Download Recording
              </Button>
            )}
            <Button
              variant="outline"
              className="border-gray-300 text-gray-900 bg-white hover:bg-gray-50 font-medium h-9 px-4"
              onClick={() => setIsDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button
              className="bg-black text-white hover:bg-gray-900 font-medium h-9 px-4"
              onClick={() => setIsDialogOpen(false)}
            >
              Continue
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

