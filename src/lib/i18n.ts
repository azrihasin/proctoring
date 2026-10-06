// UI language for user-facing alerts, selected via the `lang` URL parameter
// (`?lang=ms` for Bahasa Melayu, `?lang=en` for English). Anything else —
// including a missing param — falls back to English.
//
// Only on-screen text is translated. The eventType sent to the API and the
// `message` posted to the host application stay in English so those contracts
// don't change with the candidate's language.

export type Lang = 'en' | 'ms'

export function getLangFromUrl(): Lang {
  const lang = new URLSearchParams(window.location.search).get('lang')?.trim().toLowerCase()
  return lang === 'ms' ? 'ms' : 'en'
}

export type ViolationLabelKey =
  | 'potential_prohibited_object'
  | 'multiple_faces'
  | 'face_not_visible'
  | 'tab_switch'
  | 'wrong_face'
  | 'eyes_off_screen'
  | 'default'

type Strings = {
  warning: string
  dismissWarning: string
  violations: Record<ViolationLabelKey, string>
  cameraAccessError: string
  cameraAccessDenied: string
  cameraNotFound: string
  cameraFailed: string
  cameraAllowAndRefresh: string
  requestingCamera: string
}

export const STRINGS: Record<Lang, Strings> = {
  en: {
    warning: 'Warning',
    dismissWarning: 'Dismiss warning',
    violations: {
      potential_prohibited_object: 'Potential prohibited object detected',
      multiple_faces: 'Multiple Faces',
      face_not_visible: 'Face Not Visible',
      tab_switch: 'Tab Switch',
      wrong_face: 'Face Mismatch',
      eyes_off_screen: 'Looking Away',
      default: 'Violation',
    },
    cameraAccessError: 'Camera Access Error',
    cameraAccessDenied: 'Camera access denied. Please allow camera access in your browser settings.',
    cameraNotFound: 'No camera found. Please connect a camera and try again.',
    cameraFailed: 'Failed to access camera. Please check your camera permissions.',
    cameraAllowAndRefresh: 'Please allow camera access and refresh the page.',
    requestingCamera: 'Requesting camera access...',
  },
  ms: {
    warning: 'Amaran',
    dismissWarning: 'Tutup amaran',
    violations: {
      potential_prohibited_object: 'Kemungkinan objek larangan dikesan',
      multiple_faces: 'Lebih Daripada Satu Wajah',
      face_not_visible: 'Wajah Tidak Kelihatan',
      tab_switch: 'Pertukaran Tab',
      wrong_face: 'Wajah Tidak Sepadan',
      eyes_off_screen: 'Tidak Melihat Skrin',
      default: 'Pelanggaran',
    },
    cameraAccessError: 'Ralat Akses Kamera',
    cameraAccessDenied: 'Akses kamera ditolak. Sila benarkan akses kamera dalam tetapan pelayar anda.',
    cameraNotFound: 'Tiada kamera ditemui. Sila sambungkan kamera dan cuba lagi.',
    cameraFailed: 'Gagal mengakses kamera. Sila semak kebenaran kamera anda.',
    cameraAllowAndRefresh: 'Sila benarkan akses kamera dan muat semula halaman.',
    requestingCamera: 'Meminta akses kamera...',
  },
}
