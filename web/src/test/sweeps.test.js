import { describe, expect, it } from 'vitest'
import { suggestRange, suggestValues } from '../components/domain/SpaceEditor.jsx'

describe('sweep value suggestions', () => {
  it('spaces discount-like parameters by distance from 1', () => {
    expect(suggestValues({ type: 'float', default: 0.99, min: 0, max: 0.9999 })).toEqual([0.9, 0.97, 0.99])
  })
  it('scales learning rates around the default and stays in bounds', () => {
    const vals = suggestValues({ type: 'float', default: 0.0005, min: 1e-6, max: 1 })
    expect(vals).toEqual([0.00015, 0.0005, 0.0015])
  })
  it('never rounds a value past its bound', () => {
    const vals = suggestValues({ type: 'float', default: 0.5, min: 0, max: 0.9999 })
    expect(Math.max(...vals)).toBeLessThanOrEqual(0.9999)
  })
  it('halves and doubles integers', () => {
    expect(suggestValues({ type: 'int', default: 64, min: 16, max: 4096 })).toEqual([32, 64, 128])
  })
  it('offers both booleans and every choice', () => {
    expect(suggestValues({ type: 'bool', default: true })).toEqual([true, false])
    expect(suggestValues({ type: 'choice', default: 'a', choices: ['a', 'b'] })).toEqual(['a', 'b'])
  })
  it('suggests a log range for positive floats', () => {
    expect(suggestRange({ type: 'float', default: 0.001, min: 1e-6, max: 1 })).toEqual({
      min: 0.0001,
      max: 0.01,
      log: true,
    })
  })
})
