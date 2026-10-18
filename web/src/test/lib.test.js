import { describe, expect, it } from 'vitest'
import { downsample, ema, nearestIndex, niceTicks } from '../lib/chart.js'
import { fmtBytes, fmtDuration, fmtNum, metricLabel, runProgress } from '../lib/format.js'
import { paramError } from '../components/domain/ParamForm.jsx'

describe('chart helpers', () => {
  it('ema with alpha 0 is the identity', () => expect(ema([1, 2, 3], 0)).toEqual([1, 2, 3]))
  it('ema smooths toward history', () => expect(ema([0, 10], 0.5)).toEqual([0, 5]))
  it('niceTicks covers the range with round steps', () => {
    const t = niceTicks(0, 470)
    expect(t[0]).toBeLessThanOrEqual(0)
    expect(t[t.length - 1]).toBeGreaterThanOrEqual(470)
    expect(t[1] - t[0]).toBe(100)
  })
  it('downsample keeps extremes', () => {
    const pts = Array.from({ length: 5000 }, (_, x) => ({ x, y: x === 2500 ? 99 : 0 }))
    const out = downsample(pts, 200)
    expect(out.length).toBeLessThanOrEqual(220)
    expect(out.some((p) => p.y === 99)).toBe(true)
  })
  it('nearestIndex finds the closest x', () => expect(nearestIndex([{ x: 0 }, { x: 10 }, { x: 20 }], 14)).toBe(1))
})

describe('formatting', () => {
  it('formats numbers by magnitude', () => {
    expect(fmtNum(null)).toBe('—')
    expect(fmtNum(271.83)).toBe('271.8')
    expect(fmtNum(0.12345)).toBe('0.123')
  })
  it('formats bytes and durations', () => {
    expect(fmtBytes(1536)).toBe('1.5 KB')
    expect(fmtDuration(125)).toBe('2m 5s')
  })
  it('labels metric keys', () => expect(metricLabel('metric_td_error')).toBe('Td error'))
  it('uses live progress for active runs', () => {
    expect(runProgress({ status: 'running', episodes: 100, progress: { episode: 25, total_episodes: 100 } }).frac).toBe(
      0.25,
    )
  })
})

describe('parameter validation mirrors ParamSpec bounds', () => {
  const p = { type: 'int', min: 1, max: 10 }
  it('rejects out-of-range and non-integers', () => {
    expect(paramError(p, 0)).toMatch(/at least/)
    expect(paramError(p, 2.5)).toMatch(/whole/)
    expect(paramError(p, 5)).toBeNull()
  })
})
