import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AppProvider } from '../app/AppContext.jsx'
import { AppRoutes } from '../app/routes.jsx'
import { ToastProvider } from '../components/ui/Toast.jsx'

function renderAt(path) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <AppProvider>
          <ToastProvider>
            <AppRoutes />
          </ToastProvider>
        </AppProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

// One smoke test per route: it renders, reaches its loaded state, and shows no error.
const ROUTES = [
  ['/lab/foundations', /Tabular methods on small/],
  ['/lab/classic/new', /Choose the experiment/],
  ['/lab/classic/new?env=cartpole&agent=ppo', /Review and launch/],
  ['/lab/classic/runs', /dqn-cartpole/],
  ['/lab/classic/envs', /CartPole-v1/],
  ['/lab/classic/algorithms', /Suited to Classic Control/],
  ['/runs/0y7simbc', /Learning curve/],
  ['/envs/cartpole', /Which algorithms can train here/],
  ['/algorithms/dqn', /The idea/],
  ['/compare?runs=0y7simbc,85oe7lw7,ixwdsim9', /What differs/],
  ['/compare', /Make the comparison count/],
  ['/models', /No kept models yet/],
  ['/visualize', /Play episode/],
  ['/diagnostics', /What each probe checks/],
  ['/learn', /Meet the random agent/],
  ['/glossary', /Bootstrapping/],
  ['/no-such-page', /Page not found/],
]

describe('routes', () => {
  it.each(ROUTES)('%s renders', async (path, text) => {
    renderAt(path)
    expect(await screen.findAllByText(text, {}, { timeout: 4000 })).not.toHaveLength(0)
    expect(screen.queryByText(/Could not load/)).not.toBeInTheDocument()
  })

  it('redirects v2 URLs into the current lab', async () => {
    renderAt('/train')
    expect(await screen.findByText(/Choose the experiment/)).toBeInTheDocument()
  })
})
