/**
 * The brand mark: a rounded tile holding a rising reward curve that ends in a
 * goal node. Colours come from tokens so the mark follows the active lab.
 */
export function BrandMark({ size = 28, className }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} role="img" aria-label="RL Testbed">
      <rect width="24" height="24" rx="6" fill="rgb(var(--accent))" />
      <rect width="24" height="24" rx="6" fill="rgb(var(--text-inverse))" fillOpacity=".12" />
      <path
        d="M4.5 19.5h15"
        stroke="rgb(var(--accent-contrast))"
        strokeOpacity=".35"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <path
        d="M5 16.6 9.6 14.6 13.6 11 17.8 7.6"
        fill="none"
        stroke="rgb(var(--accent-contrast))"
        strokeWidth="1.9"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="18.6" cy="6.6" r="2.6" fill="rgb(var(--accent-contrast))" />
    </svg>
  )
}

export function Wordmark() {
  return (
    <span className="flex items-baseline gap-1 font-display text-title-s font-semibold tracking-tight text-text">
      RL <span className="font-medium text-muted">Testbed</span>
    </span>
  )
}
