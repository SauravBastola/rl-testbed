import katex from 'katex'
import 'katex/dist/katex.min.css'

function inline(text, keyBase) {
  // **bold**, `code`, $math$ — the only markup the specs use.
  const parts = text.split(/(\$[^$]+\$|\*\*[^*]+\*\*|`[^`]+`)/g)
  return parts.map((part, i) => {
    const key = `${keyBase}-${i}`
    if (part.startsWith('$') && part.endsWith('$') && part.length > 2) {
      const html = katex.renderToString(part.slice(1, -1), { throwOnError: false, output: 'html' })
      return <span key={key} dangerouslySetInnerHTML={{ __html: html }} />
    }
    if (part.startsWith('**') && part.endsWith('**')) return <strong key={key}>{part.slice(2, -2)}</strong>
    if (part.startsWith('`') && part.endsWith('`')) return <code key={key}>{part.slice(1, -1)}</code>
    return part
  })
}

/** Markdown-lite with KaTeX for the `docs.math` field of an algorithm spec. */
export function MathText({ source, className }) {
  if (!source) return null
  const paragraphs = source.split(/\n{2,}/).filter((p) => p.trim())
  return (
    <div className={className}>
      {paragraphs.map((p, i) => (
        <p key={i}>{inline(p.replace(/\n/g, ' '), i)}</p>
      ))}
    </div>
  )
}
