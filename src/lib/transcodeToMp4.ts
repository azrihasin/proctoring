import { FFmpeg } from '@ffmpeg/ffmpeg'
import { fetchFile, toBlobURL } from '@ffmpeg/util'
import { fixWebmMetadata } from './utils'

let ffmpegInstance: FFmpeg | null = null
let ffmpegLoading = false
let ffmpegLoaded = false

/**
 * Initialize FFmpeg instance (singleton pattern)
 * 
 * IMPORTANT: FFmpeg.wasm requires special COOP/COEP headers to work due to SharedArrayBuffer requirements.
 * Configure your server to send:
 * - Cross-Origin-Embedder-Policy: require-corp
 * - Cross-Origin-Opener-Policy: same-origin
 * 
 * For Vite dev server, you can configure this in vite.config.ts:
 * ```ts
 * export default defineConfig({
 *   server: {
 *     headers: {
 *       'Cross-Origin-Embedder-Policy': 'require-corp',
 *       'Cross-Origin-Opener-Policy': 'same-origin',
 *     },
 *   },
 * })
 * ```
 */
async function getFFmpeg(): Promise<FFmpeg> {
  if (ffmpegInstance && ffmpegLoaded) {
    return ffmpegInstance
  }

  if (ffmpegLoading) {
    // Wait for existing load to complete
    while (ffmpegLoading) {
      await new Promise(resolve => setTimeout(resolve, 100))
    }
    if (ffmpegInstance && ffmpegLoaded) {
      return ffmpegInstance
    }
  }

  ffmpegLoading = true
  try {
    const ffmpeg = new FFmpeg()
    
    const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm'
    await ffmpeg.load({
      coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
      wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
    })

    ffmpegInstance = ffmpeg
    ffmpegLoaded = true
    ffmpegLoading = false
    return ffmpeg
  } catch (error) {
    ffmpegLoading = false
    throw error
  }
}

/**
 * Transcode a video blob to MP4 format using FFmpeg.wasm
 * 
 * This function converts WebM (or other formats) to MP4 in the browser without needing a backend server.
 * It preserves video duration, frame rates, and metadata automatically.
 * 
 * Performance Note: FFmpeg.wasm can be resource-intensive for large files. 
 * It works best for shorter videos (< 5 minutes recommended).
 * 
 * @param input - Input video blob (WebM or other format)
 * @returns Promise resolving to MP4 blob with preserved duration and metadata
 * @throws Error if transcoding fails
 */
export async function transcodeToMp4(input: Blob): Promise<Blob> {
  // Check if already MP4
  if (input.type.includes('mp4')) {
    return input
  }

  try {
    // Fix WebM metadata before transcoding to ensure duration is preserved
    console.log('🔧 Fixing WebM metadata before transcoding...')
    const fixedInput = await fixWebmMetadata(input)
    
    const ffmpeg = await getFFmpeg()
    const inputFileName = 'input.webm'
    const outputFileName = 'output.mp4'

    // Write input file to FFmpeg file system
    await ffmpeg.writeFile(inputFileName, await fetchFile(fixedInput))

    // Convert to MP4 with flags that preserve duration and metadata:
    // -fflags +genpts: Generate presentation timestamps to preserve timing
    // -map 0:v:0: Map first video stream
    // -map 0:a:0?: Map first audio stream (optional, if available)
    // -c:v libx264: Use H.264 video codec
    // -preset fast: Balance between speed and compression
    // -crf 23: Constant rate factor for quality (23 is good quality)
    // -pix_fmt yuv420p: Pixel format for compatibility
    // -c:a aac: Use AAC audio codec
    // -b:a 128k: Audio bitrate
    // -movflags +faststart: Enable fast start for web playback
    // -avoid_negative_ts make_zero: Handle negative timestamps to preserve duration
    await ffmpeg.exec([
      '-fflags', '+genpts',
      '-i', inputFileName,
      '-map', '0:v:0',
      '-map', '0:a:0?',
      '-c:v', 'libx264',
      '-preset', 'fast',
      '-crf', '23',
      '-pix_fmt', 'yuv420p',
      '-c:a', 'aac',
      '-b:a', '128k',
      '-movflags', '+faststart',
      '-avoid_negative_ts', 'make_zero',
      outputFileName
    ])

    // Read the result from FFmpeg file system
    const data = await ffmpeg.readFile(outputFileName)
    
    // Sanity check: tiny output is almost certainly broken
    const uint8 = data instanceof Uint8Array ? data : new TextEncoder().encode(data)
    if (!uint8 || uint8.length < 1024) {
      throw new Error('FFmpeg output too small, conversion likely failed')
    }
    
    // Convert FileData to Blob
    const uint8Array = new Uint8Array(uint8.length)
    uint8Array.set(uint8)
    const mp4Blob = new Blob([uint8Array], { type: 'video/mp4' })

    // Clean up temporary files
    await ffmpeg.deleteFile(inputFileName)
    await ffmpeg.deleteFile(outputFileName)

    console.log('✅ Video converted to MP4 successfully with preserved duration')
    return mp4Blob
  } catch (error) {
    console.error('Error transcoding to MP4:', error)
    throw new Error(`Failed to transcode video to MP4: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}
