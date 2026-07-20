// Communicates with the host exam application this app is embedded into via <iframe>.
// The host listens with window.addEventListener('message', ...) and expects
// event.data to be an object of the form { type, eventType, message }.

const TARGET_ORIGIN = '*'

function isEmbedded(): boolean {
  return typeof window !== 'undefined' && window.parent !== window
}

/** Posts a { type, eventType, message } object to the parent window. */
export function sendEventToParent(type: string, eventType: string | null, message: string): void {
  const payload = { type, eventType, message }
  if (!isEmbedded()) {
    console.log('[parentMessenger] not embedded — skipped sending:', payload)
    return
  }
  console.log('[parentMessenger] posting to parent:', payload)
  window.parent.postMessage(payload, TARGET_ORIGIN)
}
