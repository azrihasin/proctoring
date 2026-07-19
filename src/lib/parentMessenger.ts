// Communicates with the host exam application this app is embedded into via <iframe>.
// The host listens with window.addEventListener('message', ...) and expects
// event.data to be a string of the form 'proctor-event:<type>:<message>'.

const TARGET_ORIGIN = '*'

function isEmbedded(): boolean {
  return typeof window !== 'undefined' && window.parent !== window
}

/** Low-level helper: posts a raw payload to the parent window, if embedded. */
export function sendToParent(data: unknown): void {
  if (!isEmbedded()) return
  window.parent.postMessage(data, TARGET_ORIGIN)
}

/** Posts a namespaced 'proctor-event:<type>:<message>' string to the parent window. */
export function sendEventToParent(type: string, message: string): void {
  sendToParent('proctor-event:' + type + ':' + message)
}
