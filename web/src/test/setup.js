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

/**
 * Node 22+ ships its own experimental global `localStorage`, which shadows
 * jsdom's: without --localstorage-file it is undefined (Node 25 warns
 * "localStorage is not available"). Tests then fail on whichever Node version
 * the machine has. Install an in-memory Storage when the real one is unusable.
 */
class MemoryStorage {
  #data = new Map()
  get length() {
    return this.#data.size
  }
  key(i) {
    return [...this.#data.keys()][i] ?? null
  }
  getItem(k) {
    return this.#data.has(String(k)) ? this.#data.get(String(k)) : null
  }
  setItem(k, v) {
    this.#data.set(String(k), String(v))
  }
  removeItem(k) {
    this.#data.delete(String(k))
  }
  clear() {
    this.#data.clear()
  }
}

for (const name of ['localStorage', 'sessionStorage']) {
  let usable = false
  try {
    const store = window[name]
    store.setItem('__probe__', '1')
    store.removeItem('__probe__')
    usable = true
  } catch {
    usable = false
  }
  if (!usable) {
    const store = new MemoryStorage()
    for (const target of [window, globalThis]) {
      try {
        Object.defineProperty(target, name, { value: store, configurable: true, writable: true })
      } catch {
        /* non-configurable on this runtime; the other target still gets it */
      }
    }
  }
}

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
