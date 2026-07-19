// Communicates with the host exam application this app is embedded into via <iframe>.
// The host listens with window.addEventListener('message', ...) and expects
// event.data to be a string of the form 'proctor-event:<type>:<message>'.

const TARGET_ORIGIN = '*'

function isEmbedded(): boolean {
  return typeof window !== 'undefined' && window.parent !== window
}

/** Posts a namespaced 'proctor-event:<type>:<message>' string to the parent window. */
export function sendEventToParent(type: string, message: string): void {
  if (!isEmbedded()) return
  window.parent.postMessage('proctor-event:' + type + ':' + message, TARGET_ORIGIN)
}
