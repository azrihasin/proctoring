// Communicates with the host exam application this app is embedded into via <iframe>.
// The host listens with window.addEventListener('message', ...) and expects
// event.data to be the plain string 'close-exam-modal' (see integration snippet).

const TARGET_ORIGIN = '*'

function isEmbedded(): boolean {
  return typeof window !== 'undefined' && window.parent !== window
}

/** Low-level helper: posts a raw payload to the parent window, if embedded. */
export function sendToParent(data: unknown): void {
  if (!isEmbedded()) return
  window.parent.postMessage(data, TARGET_ORIGIN)
}

/** Tells the host application to close/hide the exam modal that embeds this app. */
export function closeExamModal(): void {
  sendToParent('close-exam-modal')
}
