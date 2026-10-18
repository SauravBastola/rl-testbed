/**
 * Lab identity art. Each lab has a small glyph (navigation) and a larger
 * "specimen" (overview hero): a drawing of the kind of problem the lab studies.
 * Hand-drawn SVG, coloured only through tokens, drawn once on mount.
 */
import { cn } from '../../lib/utils.js'

const stroke = { fill: 'none', stroke: 'currentColor', strokeLinecap: 'round', strokeLinejoin: 'round' }

const GLYPHS = {
  grid: (
    <>
      <rect x="3" y="3" width="18" height="18" rx="3" {...stroke} strokeWidth="1.6" />
      <path d="M9 3v18M15 3v18M3 9h18M3 15h18" {...stroke} strokeWidth="1.2" opacity=".55" />
      <rect x="15.6" y="15.6" width="4.8" height="4.8" rx="1" fill="currentColor" />
    </>
  ),
  wave: (
    <>
      <path d="M3 12c2.2-6 4.4-6 6.6 0s4.4 6 6.6 0 3.3-4.5 4.8-3" {...stroke} strokeWidth="1.8" />
      <path d="M3 19h18" {...stroke} strokeWidth="1.2" opacity=".5" />
    </>
  ),
  joint: (
    <>
      <path d="M5 19 10 12l5 3 4-9" {...stroke} strokeWidth="1.8" />
      <circle cx="10" cy="12" r="2" fill="currentColor" />
      <circle cx="15" cy="15" r="2" fill="currentColor" />
      <circle cx="5" cy="19" r="1.6" {...stroke} strokeWidth="1.4" />
    </>
  ),
  pixels: (
    <>
      {[
        [4, 4],
        [10, 4],
        [16, 10],
        [4, 10],
        [10, 16],
        [16, 16],
      ].map(([x, y]) => (
        <rect
          key={`${x}${y}`}
          x={x}
          y={y}
          width="4.5"
          height="4.5"
          rx=".8"
          fill="currentColor"
          opacity={x === 16 && y === 10 ? 1 : 0.5}
        />
      ))}
      <rect x="3" y="3" width="18" height="18" rx="3" {...stroke} strokeWidth="1.4" opacity=".6" />
    </>
  ),
  seats: (
    <>
      <circle cx="8" cy="12" r="4" {...stroke} strokeWidth="1.8" />
      <circle cx="16" cy="12" r="4" fill="currentColor" opacity=".9" />
    </>
  ),
  pulse: <path d="M3 12h4l2-5 4 10 2-5h6" {...stroke} strokeWidth="1.8" />,
}

export function LabGlyph({ icon, size = 18, className }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden className={className}>
      {GLYPHS[icon] || GLYPHS.grid}
    </svg>
  )
}

/* ------------------------------------------------------------------ specimens */
const drawn = (len) => ({ strokeDasharray: len, '--len': len })

function GridSpecimen() {
  // 6x6 gridworld: value heatmap rising toward the goal, greedy policy arrows.
  const n = 6
  const cell = 26
  const cells = []
  for (let r = 0; r < n; r += 1) {
    for (let c = 0; c < n; c += 1) {
      const v = (r + c) / (2 * (n - 1))
      const goal = r === n - 1 && c === n - 1
      const right = c < n - 1 && (r === n - 1 || (r + c) % 2 === 0)
      cells.push(
        <g key={`${r}${c}`} transform={`translate(${c * cell} ${r * cell})`}>
          <rect
            x="1.5"
            y="1.5"
            width={cell - 3}
            height={cell - 3}
            rx="4"
            fill="rgb(var(--accent))"
            fillOpacity={goal ? 0.9 : 0.06 + v * 0.42}
          />
          {!goal &&
            (right ? (
              <path
                d={`M8 13h10m-3.5-3.5L18 13l-3.5 3.5`}
                {...stroke}
                stroke="rgb(var(--text))"
                strokeOpacity=".7"
                strokeWidth="1.4"
              />
            ) : (
              <path
                d={`M13 8v10m-3.5-3.5L13 18l3.5-3.5`}
                {...stroke}
                stroke="rgb(var(--text))"
                strokeOpacity=".7"
                strokeWidth="1.4"
              />
            ))}
          {goal && <circle cx="13" cy="13" r="4" fill="rgb(var(--accent-contrast))" />}
        </g>,
      )
    }
  }
  return (
    <svg viewBox="-4 -4 164 164" className="h-full w-full" aria-hidden>
      {cells}
    </svg>
  )
}

function PhaseSpecimen() {
  // Damped phase portrait (angle vs angular velocity) spiralling into balance.
  const pts = []
  for (let t = 0; t <= 26; t += 0.08) {
    const r = 62 * Math.exp(-0.085 * t)
    pts.push(`${(90 + r * Math.cos(t)).toFixed(1)} ${(80 + r * 0.72 * Math.sin(t)).toFixed(1)}`)
  }
  return (
    <svg viewBox="0 0 180 160" className="h-full w-full" aria-hidden>
      <path d="M10 80h160M90 12v136" stroke="rgb(var(--text-subtle))" strokeOpacity=".35" strokeWidth="1" />
      <path
        d={`M${pts.join(' L')}`}
        {...stroke}
        stroke="rgb(var(--accent))"
        strokeWidth="1.8"
        className="animate-draw"
        style={drawn(1400)}
      />
      <circle cx="90" cy="80" r="4" fill="rgb(var(--accent))" />
      <circle cx="90" cy="80" r="9" fill="none" stroke="rgb(var(--accent))" strokeOpacity=".35" />
    </svg>
  )
}

function TorqueSpecimen() {
  // Three joint-torque traces, phase-shifted like a gait cycle.
  const trace = (phase, amp, y) => {
    const pts = []
    for (let x = 0; x <= 170; x += 3) {
      const v = Math.sin(x / 17 + phase) + 0.35 * Math.sin(x / 7 + phase * 2)
      pts.push(`${x + 5} ${(y + v * amp).toFixed(1)}`)
    }
    return `M${pts.join(' L')}`
  }
  return (
    <svg viewBox="0 0 180 160" className="h-full w-full" aria-hidden>
      {[34, 80, 126].map((y) => (
        <path key={y} d={`M5 ${y}h170`} stroke="rgb(var(--text-subtle))" strokeOpacity=".25" strokeWidth="1" />
      ))}
      {[
        [0, 14, 34, 1],
        [2.1, 16, 80, 0.75],
        [4.2, 12, 126, 0.5],
      ].map(([ph, amp, y, op]) => (
        <path
          key={y}
          d={trace(ph, amp, y)}
          {...stroke}
          stroke="rgb(var(--accent))"
          strokeOpacity={op}
          strokeWidth="1.8"
          className="animate-draw"
          style={drawn(600)}
        />
      ))}
    </svg>
  )
}

function FramesSpecimen() {
  // A MinAtar-style 10x10 frame: paddle, ball trail, brick row.
  const on = new Set(['0,1', '0,2', '0,3', '0,4', '0,5', '0,6', '0,7', '0,8', '1,1', '1,2', '1,5', '1,6', '1,7'])
  const trail = ['6,3', '5,4', '4,5']
  const cells = []
  for (let r = 0; r < 10; r += 1) {
    for (let c = 0; c < 10; c += 1) {
      const key = `${r},${c}`
      const brick = on.has(key)
      const t = trail.indexOf(key)
      const paddle = r === 9 && c >= 3 && c <= 5
      const ball = key === '7,2'
      const op = brick ? 0.55 : paddle || ball ? 1 : t >= 0 ? 0.18 + t * 0.08 : 0.05
      cells.push(
        <rect
          key={key}
          x={c * 15 + 1}
          y={r * 15 + 1}
          width="13"
          height="13"
          rx="2"
          fill={ball || paddle ? 'rgb(var(--text))' : 'rgb(var(--accent))'}
          fillOpacity={op}
        />,
      )
    }
  }
  return (
    <svg viewBox="-2 -2 154 154" className="h-full w-full" aria-hidden>
      {cells}
    </svg>
  )
}

function BoardSpecimen() {
  // Connect-four board mid-game; the learner's discs use the accent.
  const grid = ['.......', '.......', '...a...', '..ab...', '..bab..', '.abbaa.']
  return (
    <svg viewBox="0 0 176 152" className="h-full w-full" aria-hidden>
      <rect
        x="2"
        y="2"
        width="172"
        height="148"
        rx="10"
        fill="rgb(var(--surface-3))"
        stroke="rgb(var(--border-strong))"
      />
      {grid.flatMap((row, r) =>
        [...row].map((ch, c) => (
          <circle
            key={`${r}${c}`}
            cx={16 + c * 24}
            cy={16 + r * 24}
            r="9"
            fill={ch === 'a' ? 'rgb(var(--accent))' : ch === 'b' ? 'rgb(var(--text-muted))' : 'rgb(var(--bg))'}
            fillOpacity={ch === '.' ? 0.7 : 1}
          />
        )),
      )}
      <path d="M40 112 112 40" stroke="rgb(var(--accent))" strokeWidth="3" strokeLinecap="round" strokeOpacity=".5" />
    </svg>
  )
}

function ProbeSpecimen() {
  return (
    <svg viewBox="0 0 180 160" className="h-full w-full" aria-hidden>
      <circle cx="50" cy="80" r="22" {...stroke} stroke="rgb(var(--accent))" strokeWidth="1.8" />
      <circle cx="130" cy="80" r="22" fill="rgb(var(--accent))" fillOpacity=".85" />
      <path d="M74 72q16-14 32 0" {...stroke} stroke="rgb(var(--text-muted))" strokeWidth="1.4" />
      <path d="m102 66 4 6-7 1" {...stroke} stroke="rgb(var(--text-muted))" strokeWidth="1.4" />
      <text x="90" y="56" textAnchor="middle" className="num fill-muted text-micro">
        +1
      </text>
    </svg>
  )
}

function TextSpecimen() {
  // A prompt and a sampled answer, each character shaded by its probability.
  const rows = [
    {
      prompt: '37+45=',
      answer: [
        ['0', 0.95],
        ['8', 0.9],
        ['2', 0.6],
      ],
      ok: true,
    },
    {
      prompt: '12+09=',
      answer: [
        ['0', 0.97],
        ['2', 0.5],
        ['1', 0.85],
      ],
      ok: true,
    },
    {
      prompt: '58+76=',
      answer: [
        ['1', 0.9],
        ['3', 0.35],
        ['3', 0.55],
      ],
      ok: false,
    },
  ]
  return (
    <svg viewBox="0 0 180 160" className="h-full w-full" aria-hidden>
      {rows.map((r, i) => (
        <g key={i} transform={`translate(10 ${24 + i * 44})`}>
          <text x="0" y="16" className="num" fontSize="15" fill="rgb(var(--text-muted))">
            {r.prompt}
          </text>
          {r.answer.map(([c, p], k) => (
            <g key={k} transform={`translate(${78 + k * 22} 0)`}>
              <rect width="18" height="22" rx="4" fill="rgb(var(--accent))" fillOpacity={p * 0.55} />
              <text x="9" y="16" textAnchor="middle" className="num" fontSize="14" fill="rgb(var(--text))">
                {c}
              </text>
            </g>
          ))}
          <text x="150" y="16" fontSize="14" fill={r.ok ? 'rgb(var(--success))' : 'rgb(var(--danger))'}>
            {r.ok ? '✓' : '✗'}
          </text>
        </g>
      ))}
    </svg>
  )
}

const SPECIMENS = {
  text: TextSpecimen,
  grid: GridSpecimen,
  phase: PhaseSpecimen,
  torque: TorqueSpecimen,
  frames: FramesSpecimen,
  board: BoardSpecimen,
  probe: ProbeSpecimen,
}

export function LabSpecimen({ visual, className }) {
  const Comp = SPECIMENS[visual] || GridSpecimen
  return (
    <div className={cn('text-accent', className)}>
      <Comp />
    </div>
  )
}
