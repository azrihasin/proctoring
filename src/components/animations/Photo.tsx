import { motion } from 'framer-motion'
import { type ReactNode } from 'react'

interface PhotoProps {
  children: ReactNode
  className?: string
}

export function Photo({ children, className = '' }: PhotoProps) {
  return (
    <motion.div
      initial={{ y: 100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      exit={{ y: 100, opacity: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className={className}
    >
      {children}
    </motion.div>
  )
}
