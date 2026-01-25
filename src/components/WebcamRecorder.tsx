import { useRef, useState, useEffect, useCallback } from 'react'
import { transcodeToMp4 } from '@/lib/transcodeToMp4'

type RecordingState = 'idle' | 'ready' | 'capturing' | 'processing' | 'complete'

const BUFFER_DURATION_MS = 10000 // 10 seconds
const CAPTURE_DURATION_MS = 10000 // 10 seconds after capture

export default function WebcamRecorder() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  
  // Rolling buffer for 10 seconds before capture
  const rollingBufferRef = useRef<Array<{ blob: Blob; timestamp: number }>>([])
  const rollingBufferRecorderRef = useRef<MediaRecorder | null>(null)
  
  // Capture recording (10 seconds after trigger)
  const captureRecorderRef = useRef<MediaRecorder | null>(null)
  const captureChunksRef = useRef<Blob[]>([])
  const captureTimeoutRef = useRef<number | null>(null)

  const [state, setState] = useState<RecordingState>('idle')
  const [error, setError] = useState<string | null>(null)
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null)
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null)
  const [mimeType, setMimeType] = useState<string>('')
  const [supportsMp4, setSupportsMp4] = useState<boolean>(false)

  // Detect best mimeType and MP4 support
  useEffect(() => {
    const mimeTypes = [
      'video/mp4; codecs="avc1.424028, mp4a.40.2"',
      'video/mp4',
      'video/webm; codecs="vp9,opus"',
      'video/webm; codecs="vp8,opus"',
      'video/webm'
    ]

    const mp4Types = [
      'video/mp4; codecs="avc1.424028, mp4a.40.2"',
      'video/mp4'
    ]

    let selectedMimeType = ''
    let mp4Supported = false

    for (const mimeType of mimeTypes) {
      if (MediaRecorder.isTypeSupported(mimeType)) {
        selectedMimeType = mimeType
        if (mp4Types.includes(mimeType)) {
          mp4Supported = true
        }
        break
      }
    }

    setMimeType(selectedMimeType)
    setSupportsMp4(mp4Supported)
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanup()
    }
  }, [])

  const cleanup = useCallback(() => {
    // Stop rolling buffer recorder
    if (rollingBufferRecorderRef.current && rollingBufferRecorderRef.current.state !== 'inactive') {
      try {
        rollingBufferRecorderRef.current.stop()
      } catch (e) {
        // Ignore errors during cleanup
      }
      rollingBufferRecorderRef.current = null
    }

    // Stop capture recorder
    if (captureRecorderRef.current && captureRecorderRef.current.state !== 'inactive') {
      try {
        captureRecorderRef.current.stop()
      } catch (e) {
        // Ignore errors during cleanup
      }
      captureRecorderRef.current = null
    }

    // Clear capture timeout
    if (captureTimeoutRef.current) {
      clearTimeout(captureTimeoutRef.current)
      captureTimeoutRef.current = null
    }

    // Stop all tracks
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop())
      mediaStreamRef.current = null
    }

    // Revoke object URL
    if (recordedUrl) {
      URL.revokeObjectURL(recordedUrl)
      setRecordedUrl(null)
    }

    rollingBufferRef.current = []
    captureChunksRef.current = []
  }, [recordedUrl])

  // Start rolling buffer recorder (maintains last 10 seconds)
  const startRollingBuffer = useCallback((stream: MediaStream) => {
    try {
      if (!mimeType) {
        console.error('No mimeType available for rolling buffer')
        return
      }

      const options: MediaRecorderOptions = {
        mimeType,
        videoBitsPerSecond: 1500000
      }

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
      console.log('✅ Rolling buffer recorder started')
    } catch (error) {
      console.error('Error starting rolling buffer recorder:', error)
    }
  }, [mimeType])

  // Initialize webcam and start rolling buffer
  const initializeWebcam = useCallback(async () => {
    try {
      setError(null)
      setRecordedBlob(null)
      if (recordedUrl) {
        URL.revokeObjectURL(recordedUrl)
        setRecordedUrl(null)
      }

      // Get user media
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: true
      })

      mediaStreamRef.current = stream

      // Set video source
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
      }

      // Start rolling buffer
      if (mimeType) {
        startRollingBuffer(stream)
      }

      setState('ready')
    } catch (err) {
      console.error('Error initializing webcam:', err)
      if (err instanceof Error) {
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          setError('Camera/microphone permission denied. Please allow access and try again.')
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
          setError('No camera/microphone found. Please connect a device and try again.')
        } else {
          setError(`Failed to access camera: ${err.message}`)
        }
      } else {
        setError('Failed to access camera: Unknown error')
      }
      setState('idle')
      cleanup()
    }
  }, [mimeType, startRollingBuffer, cleanup, recordedUrl])

  // Capture 10 seconds before and after (triggered by button)
  const captureRecording = useCallback(() => {
    if (!mediaStreamRef.current || !mimeType) {
      setError('Webcam not initialized')
      return
    }

    // Stop any existing capture recording
    if (captureRecorderRef.current && captureRecorderRef.current.state !== 'inactive') {
      captureRecorderRef.current.stop()
    }

    if (captureTimeoutRef.current) {
      clearTimeout(captureTimeoutRef.current)
    }

    captureChunksRef.current = []
    setState('capturing')

    try {
      const options: MediaRecorderOptions = {
        mimeType,
        videoBitsPerSecond: 1500000
      }

      const recorder = new MediaRecorder(mediaStreamRef.current, options)
      captureRecorderRef.current = recorder

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          captureChunksRef.current.push(event.data)
        }
      }

      recorder.onerror = (event) => {
        console.error('Capture recorder error:', event)
        setError('Recording error occurred')
        setState('ready')
      }

      recorder.onstop = async () => {
        // Combine rolling buffer (10s before) + capture recording (10s after)
        const bufferChunks = rollingBufferRef.current.map(item => item.blob)
        const allChunks = [...bufferChunks, ...captureChunksRef.current]
        
        if (allChunks.length === 0) {
          setError('No recording data available')
          setState('ready')
          return
        }

        const blob = new Blob(allChunks, { type: mimeType })
        
        // Convert to MP4
        setState('processing')
        try {
          const mp4Blob = await transcodeToMp4(blob)
          const url = URL.createObjectURL(mp4Blob)
          setRecordedBlob(mp4Blob)
          setRecordedUrl(url)
          setState('complete')
        } catch (err) {
          console.error('Transcoding error:', err)
          setError(`Failed to convert to MP4: ${err instanceof Error ? err.message : 'Unknown error'}`)
          setState('ready')
        }
        
        captureChunksRef.current = []
      }

      // Start recording
      recorder.start(1000) // 1 second time slicing
      console.log('✅ Capture recording started (will record for 10 seconds)')
      
      // Stop recording after 10 seconds
      captureTimeoutRef.current = window.setTimeout(() => {
        if (captureRecorderRef.current && captureRecorderRef.current.state !== 'inactive') {
          captureRecorderRef.current.stop()
        }
      }, CAPTURE_DURATION_MS)
    } catch (error) {
      console.error('Error starting capture recording:', error)
      setError('Failed to start capture recording')
      setState('ready')
    }
  }, [mimeType])


  const downloadRecording = useCallback(() => {
    if (!recordedBlob) return

    const url = recordedUrl || URL.createObjectURL(recordedBlob)
    const a = document.createElement('a')
    a.href = url
    a.download = `recording_${new Date().toISOString().replace(/[:.]/g, '-')}.mp4`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }, [recordedBlob, recordedUrl])

  const reset = useCallback(() => {
    cleanup()
    setState('idle')
    setError(null)
    setRecordedBlob(null)
    if (recordedUrl) {
      URL.revokeObjectURL(recordedUrl)
      setRecordedUrl(null)
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
  }, [cleanup, recordedUrl])

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '20px' }}>
      <h2 style={{ marginBottom: '20px', fontSize: '24px', fontWeight: 'bold' }}>
        Webcam Recorder
      </h2>

      {/* Info Display */}
      <div style={{ 
        marginBottom: '20px', 
        padding: '12px', 
        backgroundColor: '#f5f5f5', 
        borderRadius: '8px',
        fontSize: '14px'
      }}>
        <div><strong>MIME Type:</strong> {mimeType || 'Detecting...'}</div>
        <div><strong>MP4 Direct Record:</strong> {supportsMp4 ? '✅ Supported' : '❌ Not Supported (will transcode)'}</div>
      </div>

      {/* Error Display */}
      {error && (
        <div style={{
          marginBottom: '20px',
          padding: '12px',
          backgroundColor: '#fee',
          color: '#c00',
          borderRadius: '8px',
          fontSize: '14px'
        }}>
          {error}
        </div>
      )}

      {/* Live Preview */}
      <div style={{ 
        marginBottom: '20px', 
        position: 'relative',
        backgroundColor: '#000',
        borderRadius: '8px',
        overflow: 'hidden',
        aspectRatio: '16/9'
      }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover'
          }}
        />
        {state === 'capturing' && (
          <div style={{
            position: 'absolute',
            top: '10px',
            left: '10px',
            backgroundColor: 'rgba(255, 0, 0, 0.8)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '4px',
            fontSize: '14px',
            fontWeight: 'bold',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <span style={{
              width: '12px',
              height: '12px',
              backgroundColor: 'white',
              borderRadius: '50%',
              animation: 'pulse 1s infinite'
            }}></span>
            Capturing (10s after)...
          </div>
        )}
        {state === 'ready' && (
          <div style={{
            position: 'absolute',
            top: '10px',
            left: '10px',
            backgroundColor: 'rgba(0, 128, 0, 0.8)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '4px',
            fontSize: '14px',
            fontWeight: 'bold'
          }}>
            Ready - Rolling buffer active (10s before)
          </div>
        )}
        {state === 'processing' && (
          <div style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            fontSize: '18px',
            fontWeight: 'bold'
          }}>
            Converting to MP4...
          </div>
        )}
      </div>

      {/* Playback */}
      {state === 'complete' && recordedUrl && (
        <div style={{ marginBottom: '20px' }}>
          <h3 style={{ marginBottom: '10px', fontSize: '18px' }}>Captured Video (10s before + 10s after):</h3>
          <video
            src={recordedUrl}
            controls
            style={{
              width: '100%',
              borderRadius: '8px',
              backgroundColor: '#000'
            }}
          />
        </div>
      )}

      {/* Controls */}
      <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        {state === 'idle' && (
          <button
            onClick={initializeWebcam}
            style={{
              padding: '12px 24px',
              fontSize: '16px',
              fontWeight: 'bold',
              backgroundColor: '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              transition: 'background-color 0.2s'
            }}
          >
            Start Webcam
          </button>
        )}

        {state === 'ready' && (
          <button
            onClick={captureRecording}
            disabled={state === 'capturing' || state === 'processing'}
            style={{
              padding: '12px 24px',
              fontSize: '16px',
              fontWeight: 'bold',
              backgroundColor: state === 'capturing' || state === 'processing' ? '#ccc' : '#f44336',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: state === 'capturing' || state === 'processing' ? 'not-allowed' : 'pointer',
              transition: 'background-color 0.2s'
            }}
          >
            Capture (10s before + 10s after)
          </button>
        )}

        <button
          onClick={downloadRecording}
          disabled={state !== 'complete' || !recordedBlob}
          style={{
            padding: '12px 24px',
            fontSize: '16px',
            fontWeight: 'bold',
            backgroundColor: state !== 'complete' || !recordedBlob ? '#ccc' : '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: state !== 'complete' || !recordedBlob ? 'not-allowed' : 'pointer',
            transition: 'background-color 0.2s'
          }}
        >
          Download MP4
        </button>

        <button
          onClick={reset}
          disabled={state === 'capturing' || state === 'processing'}
          style={{
            padding: '12px 24px',
            fontSize: '16px',
            fontWeight: 'bold',
            backgroundColor: state === 'capturing' || state === 'processing' ? '#ccc' : '#666',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: state === 'capturing' || state === 'processing' ? 'not-allowed' : 'pointer',
            transition: 'background-color 0.2s'
          }}
        >
          Reset
        </button>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  )
}
