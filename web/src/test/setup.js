import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'
import fixtures from './fixtures.json'

// Serve recorded API responses (captured from a real server) by path.
globalThis.fetch = vi.fn(async (input) => {
  const url = new URL(typeof input === 'string' ? input : input.url, 'http://localhost')
  const body = fixtures[url.pathname]
  if (body === undefined) {
    return new Response(JSON.stringify({ detail: 'not found' }), { status: 404 })
  }
  return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
})

class NoopSocket {
  close() {}
}
globalThis.WebSocket = NoopSocket
window.scrollTo = () => {}
Element.prototype.scrollIntoView = () => {}
window.matchMedia =
  window.matchMedia ||
  ((q) => ({
    matches: false,
    media: q,
    addEventListener() {},
    removeEventListener() {},
    addListener() {},
    removeListener() {},
  }))
