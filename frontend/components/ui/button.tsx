// frontend/components/ui/button.tsx
import React from 'react'

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement>

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className = '', ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={`px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 ${className}`}
        {...props}
      />
    )
  }
)

Button.displayName = 'Button'
