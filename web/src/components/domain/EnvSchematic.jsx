import { fmtNum } from '../../lib/format.js'

/*
 * Drawings of builtin environments that have no renderer of their own. The
 * backend sends their state as data (see `view()` in rltestbed/core/envs/builtin);
 * drawing it here keeps it crisp and in the current theme.
 */

const C = {
  cell: 'rgb(var(--surface-2))',
  line: 'rgb(var(--border))',
  wall: 'rgb(var(--text-subtle))',
  agent: 'rgb(var(--accent))',
  goal: 'rgb(var(--success))',
  start: 'rgb(var(--text-muted))',
  text: 'rgb(var(--text-muted))',
  danger: 'rgb(var(--danger))',
}

const same = (a, b) => a && b && a[0] === b[0] && a[1] === b[1]

function Grid({ v }) {
  const size = 44
  const W = v.cols * size
  const H = v.rows * size + (v.wind ? 26 : 0)
  const walls = new Set((v.walls || []).map((w) => `${w[0]},${w[1]}`))
  return (
    <svg
      viewBox={`-2 -2 ${W + 4} ${H + 4}`}
      className="h-auto max-h-[400px] w-full"
      role="img"
      aria-label={`Grid, agent at row ${v.agent[0] + 1}, column ${v.agent[1] + 1}`}
    >
      {Array.from({ length: v.rows }).flatMap((_, r) =>
        Array.from({ length: v.cols }).map((__, c) => {
          const wall = walls.has(`${r},${c}`)
          return (
            <rect
              key={`${r}-${c}`}
              x={c * size + 2}
              y={r * size + 2}
              width={size - 4}
              height={size - 4}
              rx={6}
              fill={wall ? C.wall : same(v.goal, [r, c]) ? C.goal : C.cell}
              fillOpacity={wall ? 0.8 : same(v.goal, [r, c]) ? 0.3 : 1}
              stroke={C.line}
            />
          )
        }),
      )}
      {v.start && !same(v.start, v.agent) && (
        <text
          x={v.start[1] * size + size / 2}
          y={v.start[0] * size + size / 2}
          dy="0.35em"
          textAnchor="middle"
          fill={C.start}
          fontSize="13"
          fontWeight="600"
        >
          S
        </text>
      )}
      <text
        x={v.goal[1] * size + size / 2}
        y={v.goal[0] * size + size / 2}
        dy="0.35em"
        textAnchor="middle"
        fill={C.goal}
        fontSize="13"
        fontWeight="700"
      >
        G
      </text>
      <circle cx={v.agent[1] * size + size / 2} cy={v.agent[0] * size + size / 2} r={size * 0.28} fill={C.agent} />
      {v.wind &&
        v.wind.map((w, c) => (
          <text key={c} x={c * size + size / 2} y={v.rows * size + 18} textAnchor="middle" fill={C.text} fontSize="12">
            {w ? `↑${w}` : '·'}
          </text>
        ))}
    </svg>
  )
}

function Catch({ v }) {
  const size = 32
  const W = v.cols * size
  const H = v.rows * size
  return (
    <svg
      viewBox={`0 0 ${W} ${H + size}`}
      className="h-auto max-h-[400px] w-full"
      role="img"
      aria-label="Catch: falling ball and paddle"
    >
      <rect width={W} height={H + size} rx={8} fill={C.cell} stroke={C.line} />
      <circle cx={v.ball[1] * size + size / 2} cy={v.ball[0] * size + size / 2} r={size * 0.32} fill={C.goal} />
      <rect x={v.paddle * size + 3} y={H + size * 0.3} width={size - 6} height={size * 0.4} rx={4} fill={C.agent} />
    </svg>
  )
}

function Chain({ v }) {
  const gap = 70
  const W = v.n * gap + 40
  return (
    <svg
      viewBox={`0 0 ${W} 150`}
      className="h-auto max-h-[260px] w-full"
      role="img"
      aria-label={`Chain of ${v.n} states, agent in state ${v.position + 1}`}
    >
      {Array.from({ length: v.n }).map((_, i) => {
        const x = 40 + i * gap
        return (
          <g key={i}>
            {i < v.n - 1 && <line x1={x + 22} x2={x + gap - 22} y1={70} y2={70} stroke={C.line} strokeWidth={2} />}
            <circle cx={x} cy={70} r={22} fill={i === v.position ? C.agent : C.cell} stroke={C.line} strokeWidth={2} />
            <text
              x={x}
              y={70}
              dy="0.35em"
              textAnchor="middle"
              fontSize="13"
              fontWeight="600"
              fill={i === v.position ? 'rgb(var(--accent-contrast))' : C.text}
            >
              {i + 1}
            </text>
          </g>
        )
      })}
      <text x={40} y={120} textAnchor="middle" fontSize="12" fill={C.text}>
        +{fmtNum(v.left_reward)} small, easy
      </text>
      <text x={40 + (v.n - 1) * gap} y={120} textAnchor="middle" fontSize="12" fill={C.goal}>
        +{fmtNum(v.right_reward)} big, far
      </text>
    </svg>
  )
}

function Bandit({ v }) {
  const vals = v.values
  const lo = Math.min(0, ...vals)
  const hi = Math.max(0, ...vals)
  const span = hi - lo || 1
  const W = 560
  const H = 240
  const bw = W / vals.length
  const y = (x) => 20 + ((hi - x) / span) * (H - 60)
  const best = vals.indexOf(Math.max(...vals))
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-auto max-h-[320px] w-full"
      role="img"
      aria-label="Bandit arms with their true mean rewards"
    >
      <line x1={0} x2={W} y1={y(0)} y2={y(0)} stroke={C.line} />
      {vals.map((val, i) => (
        <g key={i}>
          <rect
            x={i * bw + bw * 0.18}
            width={bw * 0.64}
            y={Math.min(y(val), y(0))}
            height={Math.abs(y(val) - y(0)) || 1}
            rx={4}
            fill={i === v.last ? C.agent : val < 0 ? C.danger : C.wall}
            fillOpacity={i === v.last ? 1 : 0.45}
          />
          <text
            x={i * bw + bw / 2}
            y={H - 22}
            textAnchor="middle"
            fontSize="12"
            fill={i === v.last ? C.agent : C.text}
            fontWeight={i === v.last ? 700 : 400}
          >
            {i + 1}
            {i === best ? '★' : ''}
          </text>
        </g>
      ))}
      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize="11" fill={C.text}>
        Arm (★ best). Bars are true mean rewards, hidden from the agent; highlighted is the arm just pulled.
      </text>
    </svg>
  )
}

const KINDS = { grid: Grid, catch: Catch, chain: Chain, bandit: Bandit }

export function EnvSchematic({ view }) {
  const Comp = KINDS[view?.kind]
  if (!Comp) return null
  return <Comp v={view} />
}
