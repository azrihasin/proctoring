// Backend base URL, injected at build time from .env.[mode] (see README).
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL

if (!apiBaseUrl) {
  throw new Error(`VITE_API_BASE_URL is not set for mode "${import.meta.env.MODE}"`)
}

export const API_BASE_URL = apiBaseUrl.replace(/\/+$/, '')
