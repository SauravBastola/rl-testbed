/** @type {import('tailwindcss').Config} */

/*
 * Prism v3 — every colour is a CSS custom property holding an RGB triplet, so
 * Tailwind's `/opacity` syntax works and themes swap by changing variables only.
 * See src/styles/tokens.css for values and docs/DECISIONS.md for the rationale.
 */
const token = (name) => `rgb(var(--${name}) / <alpha-value>)`

export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    // Spacing is Tailwind's 4px scale; only 1,2,3,4,6,8,12,16 are used for layout.
    fontSize: {
      micro: ['11px', { lineHeight: '16px' }],
      caption: ['12px', { lineHeight: '18px' }],
      'body-s': ['13px', { lineHeight: '20px' }],
      body: ['14px', { lineHeight: '22px' }],
      'title-s': ['16px', { lineHeight: '24px' }],
      title: ['20px', { lineHeight: '28px' }],
      display: ['30px', { lineHeight: '36px' }],
      hero: ['40px', { lineHeight: '44px' }],
    },
    borderRadius: {
      none: '0',
      sm: '6px',
      DEFAULT: '8px', // controls
      lg: '12px', // cards
      xl: '16px', // panels, modals
      full: '999px',
    },
    extend: {
      colors: {
        bg: token('bg'),
        'bg-soft': token('bg-soft'),
        surface: token('surface'),
        'surface-2': token('surface-2'),
        'surface-3': token('surface-3'),
        border: token('border'),
        'border-strong': token('border-strong'),
        text: token('text'),
        muted: token('text-muted'),
        subtle: token('text-subtle'),
        inverse: token('text-inverse'),
        accent: {
          DEFAULT: token('accent'),
          hover: token('accent-hover'),
          contrast: token('accent-contrast'),
        },
        brand: token('brand'),
        success: token('success'),
        warning: token('warning'),
        danger: token('danger'),
        info: token('info'),
        violet: token('hue-violet'),
        cyan: token('hue-cyan'),
        amber: token('hue-amber'),
        rose: token('hue-rose'),
        lime: token('hue-lime'),
        sky: token('hue-sky'),
        'chart-1': token('chart-1'),
        'chart-2': token('chart-2'),
        'chart-3': token('chart-3'),
        'chart-4': token('chart-4'),
        'chart-5': token('chart-5'),
        'chart-6': token('chart-6'),
      },
      fontFamily: {
        sans: ['"Inter Variable"', 'Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        display: ['"Space Grotesk"', '"Inter Variable"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      boxShadow: {
        raised: 'var(--shadow-raised)',
        overlay: 'var(--shadow-overlay)',
        focus: '0 0 0 3px rgb(var(--accent) / .35)',
      },
      transitionTimingFunction: { standard: 'cubic-bezier(.2,0,0,1)' },
      transitionDuration: { fast: '120ms', base: '200ms', slow: '320ms' },
      maxWidth: { page: '1400px' },
      keyframes: {
        shimmer: { '100%': { transform: 'translateX(100%)' } },
        'pulse-dot': { '0%,100%': { opacity: '1' }, '50%': { opacity: '.35' } },
        draw: { from: { strokeDashoffset: 'var(--len, 1000)' }, to: { strokeDashoffset: '0' } },
      },
      animation: {
        shimmer: 'shimmer 1.4s infinite',
        'pulse-dot': 'pulse-dot 1.6s ease-in-out infinite',
        draw: 'draw .4s cubic-bezier(.2,0,0,1) both',
      },
    },
  },
  plugins: [],
}
