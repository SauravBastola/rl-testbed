import { Compass } from 'lucide-react'
import { useLab } from '../app/AppContext.jsx'
import { ButtonLink } from '../components/ui/Button.jsx'
import { EmptyState } from '../components/ui/Feedback.jsx'

export default function NotFound({ what = 'this page' }) {
  const { labId } = useLab()
  return (
    <EmptyState
      icon={Compass}
      title={what === 'this page' ? 'Page not found' : `Could not find ${what}`}
      description="It may have been deleted, or the address is mistyped. Search with ⌘K, or go back to your lab."
      action={
        <ButtonLink to={`/lab/${labId}`} variant="primary">
          Back to the lab overview
        </ButtonLink>
      }
      className="mt-12"
    />
  )
}
