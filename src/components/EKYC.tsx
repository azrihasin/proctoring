import { useState, useRef, useCallback } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import Webcam from 'react-webcam'
import { Button } from '@/components/ui/button'
import { ArrowLeft } from 'lucide-react'
import { FlyingText } from './animations/FlyingText'
import { Photo } from './animations/Photo'
import { Loader } from './animations/Loader'

type DocumentType = 'passport' | 'id_card' | 'drivers_license' | null

type EKYCStep =
  | 'welcome'
  | 'document_capture_front'
  | 'check_document_front'
  | 'document_capture_back'
  | 'document_capture_back_camera'
  | 'check_document_back'
  | 'selfie'
  | 'selfie_capture'
  | 'check_selfie'
  | 'loading'
  | 'complete'

interface EKYCProps {
  onComplete: () => void
}

// Animation variants for page transitions
const pageVariants = {
  initial: { x: -50, opacity: 0 },
  animate: { x: 0, opacity: 1, transition: { duration: 0.2, delay: 0.2 } },
  exit: { x: -50, opacity: 0, transition: { duration: 0.2 } }
}

/**
 * Crop the webcam <video> to exactly the on-screen rectangle of cropEl,
 * correctly compensating for object-cover scaling + centering.
 *
 * baseEl should be the device container that contains both the webcam background
 * and the crop element, ensuring they share the same coordinate space.
 */
function cropVideoToElement(
  video: HTMLVideoElement,
  cropEl: HTMLElement,
  baseEl: HTMLElement,
  quality = 0.95
): string | null {
  const cropRect = cropEl.getBoundingClientRect()
  const baseRect = baseEl.getBoundingClientRect()

  // Check if elements are valid
  if (!cropRect || !baseRect) {
    console.error('❌ Invalid bounding rectangles')
    return null
  }

  let relX = cropRect.left - baseRect.left
  let relY = cropRect.top - baseRect.top
  let relW = cropRect.width
  let relH = cropRect.height

  // If element is partially outside, clamp to base bounds
  if (relX < 0) {
    relW += relX // Reduce width by the amount outside
    relX = 0
  }
  if (relY < 0) {
    relH += relY // Reduce height by the amount outside
    relY = 0
  }
  
  // Clamp width/height to not exceed base element
  if (relX + relW > baseRect.width) {
    relW = baseRect.width - relX
  }
  if (relY + relH > baseRect.height) {
    relH = baseRect.height - relY
  }

  if (
    relW <= 1 ||
    relH <= 1 ||
    !Number.isFinite(relX) ||
    !Number.isFinite(relY) ||
    !Number.isFinite(relW) ||
    !Number.isFinite(relH)
  ) {
    console.error('❌ Invalid relative coordinates after clamping:', { relX, relY, relW, relH, cropRect, baseRect })
    return null
  }

  const vW = video.videoWidth
  const vH = video.videoHeight
  const elW = baseRect.width
  const elH = baseRect.height
  if (!vW || !vH || !elW || !elH) return null

  // object-cover for the background video inside baseEl
  const scale = Math.max(elW / vW, elH / vH)
  const renderedW = vW * scale
  const renderedH = vH * scale
  const offsetX = (renderedW - elW) / 2
  const offsetY = (renderedH - elH) / 2

  let sx = (relX + offsetX) / scale
  let sy = (relY + offsetY) / scale
  let sw = relW / scale
  let sh = relH / scale

  sx = Math.max(0, Math.min(sx, vW - 1))
  sy = Math.max(0, Math.min(sy, vH - 1))
  sw = Math.max(1, Math.min(sw, vW - sx))
  sh = Math.max(1, Math.min(sh, vH - sy))

  const canvas = document.createElement('canvas')
  canvas.width = Math.round(sw)
  canvas.height = Math.round(sh)

  const ctx = canvas.getContext('2d')
  if (!ctx) return null

  ctx.drawImage(video, Math.round(sx), Math.round(sy), Math.round(sw), Math.round(sh), 0, 0, Math.round(sw), Math.round(sh))
  const dataUrl = canvas.toDataURL('image/jpeg', quality)
  return dataUrl && dataUrl !== 'data:,' ? dataUrl : null
}

export default function EKYC({ onComplete }: EKYCProps) {
  const [currentStep, setCurrentStep] = useState<EKYCStep>('welcome')
  const [, setSelectedDocument] = useState<DocumentType>(null)
  const [documentFrontImage, setDocumentFrontImage] = useState<string | null>(null)
  const [documentBackImage, setDocumentBackImage] = useState<string | null>(null)
  const [selfieImage, setSelfieImage] = useState<string | null>(null)
  const [isCapturing, setIsCapturing] = useState(false)
  const [needsBackCapture, setNeedsBackCapture] = useState(false)

  const webcamRef = useRef<Webcam>(null)
  const webcamBoxRef = useRef<HTMLDivElement>(null) // ✅ wrapper div of webcam background
  const deviceRef = useRef<HTMLDivElement>(null) // ✅ device container (phone box) - base coordinate space
  const captureAreaRef = useRef<HTMLDivElement>(null) // ✅ shared ref for capture area element

  const handleDocumentSelect = (type: DocumentType) => {
    setSelectedDocument(type)
    setNeedsBackCapture(type === 'id_card' || type === 'drivers_license')
    setCurrentStep('document_capture_front')
  }

  // ✅ ONLY captures what's inside the capture area element
  const handleCaptureDocument = useCallback(() => {
    setIsCapturing(true)

    const attemptCapture = (retries = 3) => {
      try {
        const video = webcamRef.current?.video
        if (!video) {
          console.error('Video element not found')
          setIsCapturing(false)
          return
        }

        if (!video.videoWidth || !video.videoHeight) {
          console.error('Video dimensions invalid:', { width: video.videoWidth, height: video.videoHeight })
          setIsCapturing(false)
          return
        }

        const base = deviceRef.current
        if (!base) {
          console.error('Device container missing')
          setIsCapturing(false)
          return
        }

        // Get the capture area element from the ref (re-check it each time)
        const captureAreaElement = captureAreaRef.current
        if (!captureAreaElement) {
          if (retries > 0) {
            // Retry if element ref is not set yet
            setTimeout(() => attemptCapture(retries - 1), 50)
            return
          }
          console.error('Capture area element ref is null')
          setIsCapturing(false)
          return
        }

        // Check if element is in the DOM
        if (!captureAreaElement.isConnected) {
          if (retries > 0) {
            // Retry if element is not yet connected to DOM
            setTimeout(() => attemptCapture(retries - 1), 50)
            return
          }
          console.error('Capture area element is not in the DOM')
          setIsCapturing(false)
          return
        }

        const captureRect = captureAreaElement.getBoundingClientRect()
        const baseRect = base.getBoundingClientRect()

        // Check if element is visible and has valid dimensions
        if (captureRect.width <= 0 || captureRect.height <= 0) {
          if (retries > 0) {
            // Retry after a short delay if element isn't ready yet
            setTimeout(() => attemptCapture(retries - 1), 50)
            return
          }
          console.error('Capture area has invalid dimensions after retries:', {
            width: captureRect.width,
            height: captureRect.height,
            left: captureRect.left,
            top: captureRect.top,
            computedStyle: window.getComputedStyle(captureAreaElement).display,
            parentWidth: captureAreaElement.parentElement?.getBoundingClientRect().width,
            parentHeight: captureAreaElement.parentElement?.getBoundingClientRect().height
          })
          setIsCapturing(false)
          return
        }

        // Check if element is within the base element's bounds (with some tolerance)
        const relX = captureRect.left - baseRect.left
        const relY = captureRect.top - baseRect.top
        
        // Allow some tolerance for negative coordinates (element might be slightly outside)
        // but ensure it's not completely outside
        if (relX + captureRect.width < -10 || relY + captureRect.height < -10) {
          console.error('Capture area is outside base element bounds:', {
            relX,
            relY,
            captureWidth: captureRect.width,
            captureHeight: captureRect.height,
            baseWidth: baseRect.width,
            baseHeight: baseRect.height
          })
          setIsCapturing(false)
          return
        }

        const cropped = cropVideoToElement(video, captureAreaElement, base, 0.95)

        if (!cropped) {
          console.error('Failed to crop to capture area')
          setIsCapturing(false)
          return
        }

        setDocumentFrontImage(cropped)
        setIsCapturing(false)
        setCurrentStep('check_document_front')
      } catch (error) {
        console.error('Error capturing document:', error)
        setIsCapturing(false)
      }
    }

    // Use requestAnimationFrame to ensure layout is complete, then attempt capture
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        // Double RAF ensures layout is fully complete
        setTimeout(() => attemptCapture(), 100)
      })
    })
  }, [])

  // ✅ ONLY captures what's inside the capture area element (same as front)
  const handleCaptureDocumentBack = useCallback(() => {
    setIsCapturing(true)

    const attemptCapture = (retries = 3) => {
      try {
        const video = webcamRef.current?.video
        if (!video) {
          console.error('Video element not found')
          setIsCapturing(false)
          return
        }

        if (!video.videoWidth || !video.videoHeight) {
          console.error('Video dimensions invalid:', { width: video.videoWidth, height: video.videoHeight })
          setIsCapturing(false)
          return
        }

        const base = deviceRef.current
        if (!base) {
          console.error('Device container missing')
          setIsCapturing(false)
          return
        }

        // Get the capture area element from the ref (re-check it each time)
        const captureAreaElement = captureAreaRef.current
        if (!captureAreaElement) {
          if (retries > 0) {
            // Retry if element ref is not set yet
            setTimeout(() => attemptCapture(retries - 1), 50)
            return
          }
          console.error('Capture area element ref is null')
          setIsCapturing(false)
          return
        }

        // Check if element is in the DOM
        if (!captureAreaElement.isConnected) {
          if (retries > 0) {
            // Retry if element is not yet connected to DOM
            setTimeout(() => attemptCapture(retries - 1), 50)
            return
          }
          console.error('Capture area element is not in the DOM')
          setIsCapturing(false)
          return
        }

        const captureRect = captureAreaElement.getBoundingClientRect()
        const baseRect = base.getBoundingClientRect()

        // Check if element is visible and has valid dimensions
        if (captureRect.width <= 0 || captureRect.height <= 0) {
          if (retries > 0) {
            // Retry after a short delay if element isn't ready yet
            setTimeout(() => attemptCapture(retries - 1), 50)
            return
          }
          console.error('Capture area has invalid dimensions after retries:', {
            width: captureRect.width,
            height: captureRect.height,
            left: captureRect.left,
            top: captureRect.top,
            computedStyle: window.getComputedStyle(captureAreaElement).display,
            parentWidth: captureAreaElement.parentElement?.getBoundingClientRect().width,
            parentHeight: captureAreaElement.parentElement?.getBoundingClientRect().height
          })
          setIsCapturing(false)
          return
        }

        // Check if element is within the base element's bounds (with some tolerance)
        const relX = captureRect.left - baseRect.left
        const relY = captureRect.top - baseRect.top
        
        // Allow some tolerance for negative coordinates (element might be slightly outside)
        // but ensure it's not completely outside
        if (relX + captureRect.width < -10 || relY + captureRect.height < -10) {
          console.error('Capture area is outside base element bounds:', {
            relX,
            relY,
            captureWidth: captureRect.width,
            captureHeight: captureRect.height,
            baseWidth: baseRect.width,
            baseHeight: baseRect.height
          })
          setIsCapturing(false)
          return
        }

        const cropped = cropVideoToElement(video, captureAreaElement, base, 0.95)

        if (!cropped) {
          console.error('Failed to crop to capture area')
          setIsCapturing(false)
          return
        }

        setDocumentBackImage(cropped)
        setIsCapturing(false)
        setCurrentStep('check_document_back')
      } catch (error) {
        console.error('Error capturing document back:', error)
        setIsCapturing(false)
      }
    }

    // Use requestAnimationFrame to ensure layout is complete, then attempt capture
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        // Double RAF ensures layout is fully complete
        setTimeout(() => attemptCapture(), 100)
      })
    })
  }, [])

  // ✅ Captures the whole webcam image for selfie
  const handleCaptureSelfie = useCallback(() => {
    setIsCapturing(true)
    setTimeout(() => {
      const imageSrc = webcamRef.current?.getScreenshot()
      if (imageSrc) {
        setSelfieImage(imageSrc)
        setIsCapturing(false)
        setCurrentStep('check_selfie')
      } else {
        setIsCapturing(false)
      }
    }, 500)
  }, [])

  const handleContinue = () => {
    if (currentStep === 'check_document_front') {
      if (needsBackCapture) {
        setCurrentStep('document_capture_back')
      } else {
        setCurrentStep('selfie')
      }
    } else if (currentStep === 'check_document_back') {
      setCurrentStep('selfie')
    } else if (currentStep === 'check_selfie') {
      setCurrentStep('loading')
      setTimeout(() => {
        setCurrentStep('complete')
        setTimeout(() => {
          onComplete()
        }, 2000)
      }, 3000)
    }
  }

  const handleStartBackCapture = () => {
    setCurrentStep('document_capture_back_camera')
  }

  const handleRetake = () => {
    if (currentStep === 'check_document_front') {
      setDocumentFrontImage(null)
      setCurrentStep('document_capture_front')
    } else if (currentStep === 'check_document_back') {
      setDocumentBackImage(null)
      setCurrentStep('document_capture_back_camera')
    } else if (currentStep === 'check_selfie') {
      setSelfieImage(null)
      setCurrentStep('selfie_capture')
    }
  }

  const handleBack = () => {
    if (currentStep === 'document_capture_front') {
      setCurrentStep('welcome')
    } else if (currentStep === 'check_document_front') {
      setCurrentStep('document_capture_front')
    } else if (currentStep === 'document_capture_back') {
      setCurrentStep('check_document_front')
    } else if (currentStep === 'document_capture_back_camera') {
      setCurrentStep('document_capture_back')
    } else if (currentStep === 'check_document_back') {
      setCurrentStep('document_capture_back_camera')
    } else if (currentStep === 'selfie') {
      if (needsBackCapture) {
        setCurrentStep('check_document_back')
      } else {
        setCurrentStep('check_document_front')
      }
    } else if (currentStep === 'selfie_capture') {
      setCurrentStep('selfie')
    } else if (currentStep === 'check_selfie') {
      setCurrentStep('selfie_capture')
    }
  }

  // Welcome Step Component
  const WelcomeStep = () => (
    <div className="h-full flex flex-col px-6 py-8">
      <div className="text-center space-y-6 flex-1 flex flex-col justify-center">
        <FlyingText>
          <h1 className="text-2xl font-bold text-gray-900 mb-2">Verify your identity</h1>
        </FlyingText>
        <FlyingText>
          <p className="text-gray-600 text-sm leading-relaxed">We need some information to help us confirm your identity.</p>
        </FlyingText>

        <Photo className="flex justify-center my-8">
          <div className="relative w-48 h-32">
            <div
              className="absolute left-0 top-0 w-20 h-28 bg-blue-500 rounded-lg shadow-lg flex items-center justify-center border-2 border-white"
              style={{ boxShadow: '0 0 20px rgba(59, 130, 246, 0.5)' }}
            >
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" className="text-white">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.5" fill="none" />
                <circle cx="12" cy="12" r="6" fill="currentColor" opacity="0.3" />
              </svg>
            </div>

            <div
              className="absolute left-10 top-4 w-32 h-20 bg-white border-2 border-blue-500 rounded-lg shadow-lg flex items-center z-10"
              style={{ boxShadow: '0 0 20px rgba(59, 130, 246, 0.3)' }}
            >
              <div className="w-10 h-10 bg-blue-500 rounded-full ml-2 flex items-center justify-center flex-shrink-0">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" className="text-white">
                  <path
                    d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"
                    fill="currentColor"
                  />
                </svg>
              </div>
              <div className="ml-2 space-y-1 flex-1">
                <div className="h-1.5 w-14 bg-blue-500 rounded"></div>
                <div className="h-1.5 w-10 bg-blue-500 rounded"></div>
                <div className="h-1.5 w-12 bg-blue-500 rounded"></div>
              </div>
            </div>

            <div className="absolute top-2 right-2 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center shadow-lg z-20">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" className="text-white">
                <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" fill="currentColor" />
              </svg>
            </div>
          </div>
        </Photo>

        <div className="pt-4">
          <Button
            onClick={() => handleDocumentSelect('id_card')}
            className="w-full bg-blue-500 hover:bg-blue-600 text-white font-medium py-6 text-base rounded-lg transition-all duration-100 ease-in"
          >
            Choose document type
          </Button>
        </div>

        <p className="text-gray-500 text-xs mt-4">Verifying usually takes a few seconds.</p>
      </div>
    </div>
  )

  // Document Capture Front Step Component
  const DocumentCaptureFrontStep = () => {
    return (
      <div className="h-full flex flex-col px-1 py-1 relative">
        <div className="space-y-3 flex-1 flex flex-col relative z-[20]">
          <div className="relative flex items-center z-[20]">
            <button
              onClick={handleBack}
              className="absolute left-0 p-2 hover:bg-gray-100 rounded-lg transition-all duration-100 ease-in z-[20]"
            >
              <ArrowLeft className="w-5 h-5 text-gray-700" />
            </button>
            <FlyingText className="flex-1 relative z-[20]">
              <h1 className="text-lg font-bold text-white text-center relative z-[20]">Upload ID</h1>
            </FlyingText>
          </div>

          <FlyingText className="relative z-[20]">
            <p className="text-white text-xs text-center leading-relaxed relative z-[20]">
              Place your ID inside the frame and take a picture.
              <br />
              Make sure it is not cut or has any glare.
            </p>
          </FlyingText>

          <div className="mt-2 relative z-[15] flex-1 flex items-center justify-center min-w-0">
            {/* Box-shadow overlay - transparent capture area with dark shadow around it */}
            <div
              ref={captureAreaRef}
              className="relative w-full aspect-[5/3] rounded-xl pointer-events-none min-w-[250px]"
              style={{
                boxShadow: '0px 0px 0px 1000px rgba(0, 0, 0, 0.7)',
                border: '3px solid rgba(255, 255, 255, 0.3)'
              }}
            />
          </div>

          <div className="flex justify-center pt-2 relative z-[30]">
            <button
              onClick={handleCaptureDocument}
              disabled={isCapturing}
              className="w-16 h-16 bg-white rounded-full border-2 border-gray-300 flex items-center justify-center hover:bg-gray-50 transition-all duration-100 ease-in disabled:opacity-50 relative z-[30]"
            >
              <div className="w-12 h-12 bg-white rounded-full border-2 border-gray-400"></div>
            </button>
          </div>
        </div>
      </div>
    )
  }

  // Check Document Front Step Component
  const CheckDocumentFrontStep = () => (
    <div className="h-full flex flex-col px-6 py-8">
      <div className="space-y-6 flex-1 flex flex-col">
        <div className="flex items-center">
          <button onClick={handleBack} className="mr-4 p-2 hover:bg-gray-100 rounded-lg transition-all duration-100 ease-in">
            <ArrowLeft className="w-5 h-5 text-gray-700" />
          </button>
          <FlyingText>
            <h1 className="text-xl font-bold text-gray-900">Review picture</h1>
          </FlyingText>
        </div>

        <FlyingText>
          <p className="text-gray-600 text-sm text-center leading-relaxed">Make sure the information is seen clearly, with no blur or glare.</p>
        </FlyingText>

        <div className="mt-8 relative z-[15] flex-1 flex items-center justify-center min-w-0">
          <Photo className="relative w-full aspect-[5/3] bg-gray-100 rounded-lg overflow-hidden min-w-[250px]">
            <img src={documentFrontImage!} alt="Document front" className="w-full h-full object-cover" />
          </Photo>
        </div>

        <div className="flex gap-4 pt-4">
          <Button onClick={handleRetake} variant="outline" className="flex-1 border-gray-300 text-gray-700 hover:bg-gray-50 transition-all duration-100 ease-in">
            Take again
          </Button>
          <Button onClick={handleContinue} className="flex-1 bg-blue-500 hover:bg-blue-600 text-white transition-all duration-100 ease-in">
            Looks good
          </Button>
        </div>
      </div>
    </div>
  )

  // Document Capture Back Step Component
  const DocumentCaptureBackStep = () => (
    <div className="h-full flex flex-col px-6 py-8">
      <div className="space-y-6 flex-1 flex flex-col justify-center">
        <div className="flex items-center">
          <button onClick={handleBack} className="mr-4 p-2 hover:bg-gray-100 rounded-lg transition-all duration-100 ease-in">
            <ArrowLeft className="w-5 h-5 text-gray-900" />
          </button>
          <FlyingText>
            <h1 className="text-xl font-bold text-gray-900">ID back side</h1>
          </FlyingText>
        </div>

        <FlyingText>
          <p className="text-gray-600 text-sm text-center">Please take a picture of the back side of your ID</p>
        </FlyingText>

        <Photo className="flex justify-center my-8">
          <div className="relative">
            <div
              className="w-32 h-24 bg-white border-2 border-blue-500 rounded-lg shadow-lg flex items-center relative z-10"
              style={{ boxShadow: '0 0 20px rgba(59, 130, 246, 0.3)' }}
            >
              <div className="w-12 h-12 bg-blue-500 rounded-full ml-2 flex items-center justify-center">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-white">
                  <path
                    d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"
                    fill="currentColor"
                  />
                </svg>
              </div>
              <div className="ml-3 space-y-1">
                <div className="h-2 w-16 bg-blue-500 rounded"></div>
                <div className="h-2 w-12 bg-blue-500 rounded"></div>
                <div className="h-2 w-14 bg-blue-500 rounded"></div>
              </div>
            </div>

            <motion.svg
              className="absolute -top-4 -right-4 w-20 h-20 text-blue-500"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              initial={{ rotate: 0 }}
              animate={{ rotate: 360 }}
              transition={{ duration: 1.4, repeat: Infinity, ease: 'linear' }}
            >
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" fill="none" opacity="0.3" />
              <path d="M8 12h8M12 8v8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              <path d="M12 2l2 4-2 4M12 22l-2-4 2-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </motion.svg>
          </div>
        </Photo>

        <div className="pt-4">
          <Button
            onClick={handleStartBackCapture}
            className="w-full bg-blue-500 hover:bg-blue-600 text-white font-medium py-6 text-base rounded-lg transition-all duration-100 ease-in"
          >
            Take photo
          </Button>
        </div>
      </div>
    </div>
  )

  // Document Capture Back Camera Step Component
  const DocumentCaptureBackCameraStep = () => {
    return (
      <div className="h-full flex flex-col px-1 py-1 relative">
        <div className="space-y-3 flex-1 flex flex-col relative z-[20]">
          <div className="relative flex items-center z-[20]">
            <button
              onClick={handleBack}
              className="absolute left-0 p-2 hover:bg-gray-100 rounded-lg transition-all duration-100 ease-in z-[20]"
            >
              <ArrowLeft className="w-5 h-5 text-gray-700" />
            </button>
            <FlyingText className="flex-1 relative z-[20]">
              <h1 className="text-lg font-bold text-white text-center relative z-[20]">ID Card back side</h1>
            </FlyingText>
          </div>

          <FlyingText className="relative z-[20]">
            <p className="text-white text-xs text-center leading-relaxed relative z-[20]">
              Place the back side of your ID Card inside the
              <br />
              frame and take a picture. Make sure it is not cut or
              <br />
              has any glare.
            </p>
          </FlyingText>

          <div className="mt-2 relative z-[15] flex-1 flex items-center justify-center min-w-0">
            {/* Box-shadow overlay - transparent capture area with dark shadow around it */}
            <div
              ref={captureAreaRef}
              className="relative w-full aspect-[5/3] rounded-xl pointer-events-none min-w-[250px]"
              style={{
                boxShadow: '0px 0px 0px 1000px rgba(0, 0, 0, 0.7)',
                border: '3px solid rgba(255, 255, 255, 0.3)'
              }}
            />
          </div>

          <div className="flex justify-center pt-2 relative z-[30]">
            <button
              onClick={handleCaptureDocumentBack}
              disabled={isCapturing}
              className="w-16 h-16 bg-white rounded-full border-2 border-gray-300 flex items-center justify-center hover:bg-gray-50 transition-all duration-100 ease-in disabled:opacity-50 relative z-[30]"
            >
              <div className="w-12 h-12 bg-white rounded-full border-2 border-gray-400"></div>
            </button>
          </div>
        </div>
      </div>
    )
  }

  // Check Document Back Step Component
  const CheckDocumentBackStep = () => (
    <div className="h-full flex flex-col px-6 py-8">
      <div className="space-y-6 flex-1 flex flex-col">
        <div className="flex items-center">
          <button onClick={handleBack} className="mr-4 p-2 hover:bg-gray-100 rounded-lg transition-all duration-100 ease-in">
            <ArrowLeft className="w-5 h-5 text-gray-700" />
          </button>
          <FlyingText>
            <h1 className="text-xl font-bold text-gray-900">Review picture</h1>
          </FlyingText>
        </div>

        <FlyingText>
          <p className="text-gray-600 text-sm text-center leading-relaxed">Make sure the information is seen clearly, with no blur or glare.</p>
        </FlyingText>

        <div className="mt-8 relative z-[15] flex-1 flex items-center justify-center min-w-0">
          <Photo className="relative w-full aspect-[5/3] bg-gray-100 rounded-lg overflow-hidden min-w-[250px]">
            <img src={documentBackImage!} alt="Document back" className="w-full h-full object-cover" />
          </Photo>
        </div>

        <div className="flex gap-4 pt-4">
          <Button onClick={handleRetake} variant="outline" className="flex-1 border-gray-300 text-gray-700 hover:bg-gray-50 transition-all duration-100 ease-in">
            Take again
          </Button>
          <Button onClick={handleContinue} className="flex-1 bg-blue-500 hover:bg-blue-600 text-white transition-all duration-100 ease-in">
            Looks good
          </Button>
        </div>
      </div>
    </div>
  )

  // Selfie Step Component
  const SelfieStep = () => (
    <div className="h-full flex flex-col px-6 py-8">
      <div className="space-y-6 flex-1 flex flex-col justify-center">
        <div className="flex items-center">
          <button onClick={handleBack} className="mr-4 p-2 hover:bg-gray-100 rounded-lg transition-all duration-100 ease-in">
            <ArrowLeft className="w-5 h-5 text-gray-900" />
          </button>
          <FlyingText>
            <h1 className="text-xl font-bold text-gray-900">Selfie</h1>
          </FlyingText>
        </div>

        <FlyingText>
          <p className="text-gray-600 text-sm text-center">Make sure your face is clear and is fully inside the frame.</p>
        </FlyingText>

        <Photo className="flex justify-center my-8">
          <div
            className="relative w-40 h-56 bg-white border-2 border-blue-500 rounded-2xl shadow-lg flex flex-col items-center justify-center p-6"
            style={{ boxShadow: '0 0 20px rgba(59, 130, 246, 0.3)' }}
          >
            <div className="w-full h-full bg-blue-50 rounded-xl flex flex-col items-center justify-center space-y-3">
              <div className="w-20 h-20 bg-blue-200 rounded-full flex items-center justify-center">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" className="text-blue-500">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.5" fill="none" />
                  <circle cx="9" cy="9" r="1.5" fill="currentColor" />
                  <circle cx="15" cy="9" r="1.5" fill="currentColor" />
                  <path d="M8 14c0 2 1.5 3 4 3s4-1 4-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
              </div>
              <div className="w-24 h-12 bg-blue-200 rounded-lg"></div>
            </div>
          </div>
        </Photo>

        <div className="pt-4">
          <Button
            onClick={() => setCurrentStep('selfie_capture')}
            className="w-full bg-blue-500 hover:bg-blue-600 text-white font-medium py-6 text-base rounded-lg transition-all duration-100 ease-in"
          >
            Take a selfie
          </Button>
        </div>
      </div>
    </div>
  )

  // Selfie Capture Step Component
  const SelfieCaptureStep = () => {
    return (
      <div className="h-full flex flex-col px-1 py-1 relative">
        <div className="space-y-3 flex-1 flex flex-col relative z-[20]">
          <div className="relative flex items-center z-[20]">
            <button
              onClick={handleBack}
              className="absolute left-0 p-2 hover:bg-gray-100 rounded-lg transition-all duration-100 ease-in z-[20]"
            >
              <ArrowLeft className="w-5 h-5 text-gray-700" />
            </button>
            <FlyingText className="flex-1 relative z-[20]">
              <h1 className="text-lg font-bold text-white text-center relative z-[20]">Selfie</h1>
            </FlyingText>
          </div>

          <FlyingText className="relative z-[20]">
            <p className="text-white text-xs text-center leading-relaxed relative z-[20]">
              Place your face inside the frame and take a picture.
              <br />
              Make sure it is clearly visible.
            </p>
          </FlyingText>

          <div className="mt-2 relative z-[15] flex-1 flex items-center justify-center min-w-0">
            {/* Box-shadow overlay - oval/circular transparent capture area with dark shadow around it */}
            <div
              ref={captureAreaRef}
              className="relative w-[75%] aspect-[3/4] rounded-full pointer-events-none max-h-[75%] min-w-[200px]"
              style={{
                boxShadow: '0px 0px 0px 1000px rgba(0, 0, 0, 0.7)',
                border: '3px solid rgba(255, 255, 255, 0.3)'
              }}
            />
          </div>

          <div className="flex justify-center pt-2 relative z-[30]">
            <button
              onClick={handleCaptureSelfie}
              disabled={isCapturing}
              className="w-16 h-16 bg-white rounded-full border-2 border-gray-300 flex items-center justify-center hover:bg-gray-50 transition-all duration-100 ease-in disabled:opacity-50 relative z-[30]"
            >
              <div className="w-12 h-12 bg-white rounded-full border-2 border-gray-400"></div>
            </button>
          </div>
        </div>
      </div>
    )
  }

  // Check Selfie Step Component
  const CheckSelfieStep = () => (
    <div className="h-full flex flex-col px-6 py-8">
      <div className="space-y-6 flex-1 flex flex-col">
        <div className="flex items-center">
          <button onClick={handleBack} className="mr-4 p-2 hover:bg-gray-100 rounded-lg transition-all duration-100 ease-in">
            <ArrowLeft className="w-5 h-5 text-gray-700" />
          </button>
          <FlyingText>
            <h1 className="text-xl font-bold text-gray-900">Review picture</h1>
          </FlyingText>
        </div>

        <FlyingText>
          <p className="text-gray-600 text-sm text-center leading-relaxed">Make sure the information is seen clearly, with no blur or glare.</p>
        </FlyingText>

        <div className="mt-8 relative z-[15] flex-1 flex items-center justify-center min-w-0">
          <Photo className="relative w-full aspect-square max-w-md mx-auto bg-gray-100 rounded-lg overflow-hidden min-w-[250px]">
            <img src={selfieImage!} alt="Selfie" className="w-full h-full object-cover" />
          </Photo>
        </div>

        <div className="flex gap-4 pt-4">
          <Button onClick={handleRetake} variant="outline" className="flex-1 border-gray-300 text-gray-700 hover:bg-gray-50 transition-all duration-100 ease-in">
            Take again
          </Button>
          <Button onClick={handleContinue} className="flex-1 bg-blue-500 hover:bg-blue-600 text-white transition-all duration-100 ease-in">
            Looks good
          </Button>
        </div>
      </div>
    </div>
  )

  // Loading Step Component
  const LoadingStep = () => (
    <div className="h-full flex flex-col px-6 py-8">
      <div className="text-center space-y-6 flex-1 flex flex-col justify-center">
        <FlyingText>
          <h1 className="text-xl font-bold text-gray-900 leading-relaxed">We're verifying your<br />documents</h1>
        </FlyingText>

        <div className="flex justify-center py-8">
          <Loader size={48} />
        </div>
      </div>
    </div>
  )

  const stepComponents: Record<EKYCStep, React.ComponentType<any>> = {
    welcome: WelcomeStep,
    document_capture_front: DocumentCaptureFrontStep,
    check_document_front: CheckDocumentFrontStep,
    document_capture_back: DocumentCaptureBackStep,
    document_capture_back_camera: DocumentCaptureBackCameraStep,
    check_document_back: CheckDocumentBackStep,
    selfie: SelfieStep,
    selfie_capture: SelfieCaptureStep,
    check_selfie: CheckSelfieStep,
    loading: LoadingStep,
    complete: LoadingStep
  }

  const CurrentStepComponent = stepComponents[currentStep]

  const isDarkBackground =
    currentStep === 'document_capture_front' || currentStep === 'document_capture_back_camera' || currentStep === 'selfie_capture'

  const showWebcamAsBackground = currentStep === 'document_capture_front' || currentStep === 'document_capture_back_camera' || currentStep === 'selfie_capture'
  const useFrontCamera = currentStep === 'selfie_capture'

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center overflow-hidden"
      style={{ background: 'linear-gradient(180deg, #fff 0%, #fff 75%, #007AFF 250%)' }}
    >
      <div
        ref={deviceRef}
        className={`relative transition-all duration-100 ease-in ${
          showWebcamAsBackground ? 'bg-black' : isDarkBackground ? 'bg-gray-800' : 'bg-white'
        }`}
        style={{
          width: '415px',
          height: '660px',
          boxShadow: '0 20px 40px rgba(0, 0, 0, 0.05)',
          borderRadius: '35px',
          overflow: 'hidden',
          padding: '40px'
        }}
      >
        {showWebcamAsBackground && (
          <div ref={webcamBoxRef} className="absolute inset-0">
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              videoConstraints={{
                facingMode: useFrontCamera ? 'user' : 'environment',
                width: { ideal: useFrontCamera ? 640 : 1280 },
                height: { ideal: useFrontCamera ? 640 : 720 }
              }}
              className="w-full h-full object-cover outline-none"
              mirrored={useFrontCamera}
            />
            <div className="absolute inset-0 bg-black/35" />
          </div>
        )}

        <AnimatePresence mode="wait">
          <motion.div
            key={currentStep}
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="relative z-10 w-full h-full"
          >
            <CurrentStepComponent />
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}
