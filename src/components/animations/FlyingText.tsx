import { motion } from 'framer-motion'
import { ReactNode } from 'react'

interface FlyingTextProps {
  children: ReactNode
  className?: string
}

export function FlyingText({ children, className = '' }: FlyingTextProps) {
  return (
    <motion.div
      key={children?.toString()}
      initial={{ y: 16, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      exit={{ y: -16, opacity: 0 }}
      transition={{ duration: 0.2, ease: 'easeOut' }}
      className={className}
    >
      {children}
    </motion.div>
  )
}
