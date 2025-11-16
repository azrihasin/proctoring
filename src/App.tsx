import { useEffect, useRef, useState, useCallback } from 'react'
import Webcam from 'react-webcam'
import * as tf from '@tensorflow/tfjs'
import * as cocoSsd from '@tensorflow-models/coco-ssd'
import * as blazeface from '@tensorflow-models/blazeface'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { AlertTriangle } from 'lucide-react'

type DetectionType = 'cell_phone' | 'prohibited_object' | 'face_not_visible' | null

function App() {
  const webcamRef = useRef<Webcam>(null)
  const [detection, setDetection] = useState<DetectionType>(null)
  const [isDialogOpen, setIsDialogOpen] = useState(false)
  const [detectionMessage, setDetectionMessage] = useState('')
  const cocoModelRef = useRef<cocoSsd.ObjectDetection | null>(null)
  const faceModelRef = useRef<blazeface.BlazeFaceModel | null>(null)
  const detectionIntervalRef = useRef<number | null>(null)
  const [modelsLoaded, setModelsLoaded] = useState(false)

  // Prohibited objects (excluding cell phone which is handled separately)
  const prohibitedObjects = ['book', 'laptop', 'keyboard', 'mouse', 'remote', 'tv']

  // Initialize TensorFlow models
  useEffect(() => {
    const loadModels = async () => {
      try {
        await tf.ready()
        
        // Load COCO-SSD model for object detection
        const cocoModel = await cocoSsd.load()
        cocoModelRef.current = cocoModel
        
        // Load BlazeFace model for face detection
        const faceModel = await blazeface.load()
        faceModelRef.current = faceModel
        
        setModelsLoaded(true)
        console.log('Models loaded successfully')
      } catch (error) {
        console.error('Error loading models:', error)
      }
    }

    loadModels()

    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current)
      }
    }
  }, [])

  // Detection function
  const detectCheating = useCallback(async () => {
    if (!webcamRef.current || !cocoModelRef.current || !faceModelRef.current) {
      return
    }

    const video = webcamRef.current.video
    if (!video || video.readyState !== video.HAVE_ENOUGH_DATA) {
      return
    }

    try {
      // Detect objects using COCO-SSD
      const objects = await cocoModelRef.current.detect(video)
      
      // Detect faces using BlazeFace
      const faces = await faceModelRef.current.estimateFaces(video, false)

      let detectedType: DetectionType = null
      let message = ''

      // Check for cell phone
      const cellPhone = objects.find(
        obj => obj.class === 'cell phone' && obj.score > 0.5
      )
      if (cellPhone) {
        detectedType = 'cell_phone'
        message = 'Cell phone detected! Please remove any mobile devices from the exam area.'
      }

      // Check for prohibited objects
      if (!detectedType) {
        const prohibited = objects.find(
          obj => prohibitedObjects.includes(obj.class) && obj.score > 0.5
        )
        if (prohibited) {
          detectedType = 'prohibited_object'
          message = `Prohibited object detected: ${prohibited.class}. Please remove all unauthorized items from your workspace.`
        }
      }

      // Check for face visibility
      if (!detectedType) {
        if (faces.length === 0) {
          detectedType = 'face_not_visible'
          message = 'Face not visible! Please ensure your face is clearly visible in the camera frame.'
        }
      }

      // Show dialog if detection found
      if (detectedType) {
        setDetection(detectedType)
        setDetectionMessage(message)
        setIsDialogOpen(true)
      } else {
        setDetection(null)
      }
    } catch (error) {
      console.error('Error during detection:', error)
    }
  }, [prohibitedObjects])

  // Start detection loop
  useEffect(() => {
    if (modelsLoaded && cocoModelRef.current && faceModelRef.current) {
      // Run detection every 2 seconds
      detectionIntervalRef.current = window.setInterval(() => {
        detectCheating()
      }, 2000)
    }

    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current)
      }
    }
  }, [modelsLoaded, detectCheating])

  const getDetectionTitle = () => {
    switch (detection) {
      case 'cell_phone':
        return 'Cell Phone Detected'
      case 'prohibited_object':
        return 'Prohibited Object Detected'
      case 'face_not_visible':
        return 'Face Not Visible'
      default:
        return 'Detection Alert'
    }
  }

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="w-full max-w-4xl">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h1 className="text-2xl font-bold text-center mb-6 text-gray-800">
            Proctoring System
          </h1>
          <div className="flex justify-center">
            <div className="relative">
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                videoConstraints={{
                  width: 1280,
                  height: 720,
                  facingMode: 'user',
                }}
                className="rounded-lg border-4 border-gray-300"
              />
              {detection && (
                <div className="absolute top-2 right-2 bg-red-500 text-white px-3 py-1 rounded-full text-sm font-semibold flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" />
                  Alert
                </div>
              )}
            </div>
          </div>
          {!modelsLoaded && (
            <p className="text-center mt-4 text-blue-600 font-semibold">
              Loading detection models...
            </p>
          )}
          {modelsLoaded && (
            <p className="text-center mt-4 text-gray-600">
              Please keep your face visible and remove any prohibited items from your workspace.
            </p>
          )}
        </div>
      </div>

      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-red-600">
              <AlertTriangle className="w-5 h-5" />
              {getDetectionTitle()}
            </DialogTitle>
            <DialogDescription className="pt-4 text-base">
              {detectionMessage}
            </DialogDescription>
          </DialogHeader>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default App

