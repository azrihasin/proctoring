import { motion } from 'framer-motion'

interface LoaderProps {
  className?: string
  size?: number
}

export function Loader({ className = '', size = 40 }: LoaderProps) {
  return (
    <motion.div
      className={className}
      style={{ width: size, height: size }}
      animate={{ rotate: 360 }}
      transition={{ duration: 0.7, repeat: Infinity, ease: 'linear' }}
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 40 40"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <circle
          cx="20"
          cy="20"
          r="18"
          stroke="#3B82F6"
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray="56.55"
          strokeDashoffset="14.14"
        />
      </svg>
    </motion.div>
  )
}
