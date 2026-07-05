import { useEffect, useState } from "react"

import { cn } from "@/lib/utils"

/**
 * Renders a playable inline preview for a recorded video Blob.
 * Manages the object URL lifecycle (created on mount / blob change, revoked on
 * cleanup) so the browser doesn't leak blob URLs as clips are added/removed.
 */
export function VideoPreview({
  blob,
  className,
}: {
  blob: Blob
  className?: string
}) {
  const [url, setUrl] = useState<string | null>(null)

  useEffect(() => {
    const objectUrl = URL.createObjectURL(blob)
    setUrl(objectUrl)
    return () => URL.revokeObjectURL(objectUrl)
  }, [blob])

  if (!url) return null

  return (
    <video
      src={url}
      controls
      preload="metadata"
      playsInline
      className={cn(
        "w-full aspect-video rounded-md bg-black object-cover",
        className
      )}
    />
  )
}
