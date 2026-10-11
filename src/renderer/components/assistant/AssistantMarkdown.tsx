import { Fragment, type ReactNode } from 'react'

// A small Markdown subset for assistant replies: paragraphs, headings, lists,
// code, bold, italics and inline code. Everything becomes React text nodes,
// never HTML, and links show as plain text: replies can quote video titles and
// transcripts, so nothing in them becomes clickable or executable.

function inline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = []
  const pattern = /(`[^`\n]+`|\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|\[[^\]\n]+\]\([^)\s]+\))/g
  let last = 0
  let index = 0
  for (const match of text.matchAll(pattern)) {
    const at = match.index ?? 0
    if (at > last) nodes.push(text.slice(last, at))
    const token = match[0]
    const key = `${keyPrefix}-${index++}`
    if (token.startsWith('`')) nodes.push(<code key={key} className="rounded-md bg-white/[0.07] px-1 py-px font-mono text-[12px] text-ink">{token.slice(1, -1)}</code>)
    else if (token.startsWith('**') || token.startsWith('__')) nodes.push(<strong key={key} className="font-semibold text-ink">{token.slice(2, -2)}</strong>)
    else if (token.startsWith('[')) {
      const label = token.slice(1, token.indexOf(']'))
      const url = token.slice(token.indexOf('(') + 1, -1)
      nodes.push(<Fragment key={key}>{label}{label !== url && <span className="text-ink-subtle"> ({url})</span>}</Fragment>)
    } else nodes.push(<em key={key}>{token.slice(1, -1)}</em>)
    last = at + token.length
  }
  if (last < text.length) nodes.push(text.slice(last))
  return nodes
}

type Block =
  | { kind: 'paragraph'; lines: string[] }
  | { kind: 'heading'; level: number; text: string }
  | { kind: 'list'; ordered: boolean; items: string[] }
  | { kind: 'code'; text: string }

function blocks(source: string): Block[] {
  const result: Block[] = []
  const lines = source.replace(/\r\n/g, '\n').split('\n')
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (/^\s*```/.test(line)) {
      const body: string[] = []
      i++
      while (i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i++])
      i++
      result.push({ kind: 'code', text: body.join('\n') })
      continue
    }
    const heading = /^(#{1,4})\s+(.*)$/.exec(line)
    if (heading) {
      result.push({ kind: 'heading', level: heading[1].length, text: heading[2] })
      i++
      continue
    }
    const bullet = /^\s*([-*•]|\d+[.)])\s+(.*)$/.exec(line)
    if (bullet) {
      const ordered = /\d/.test(bullet[1])
      const items: string[] = []
      while (i < lines.length) {
        const item = /^\s*([-*•]|\d+[.)])\s+(.*)$/.exec(lines[i])
        if (item && /\d/.test(item[1]) === ordered) { items.push(item[2]); i++ }
        else if (lines[i].trim() && /^\s{2,}/.test(lines[i]) && items.length) { items[items.length - 1] += ` ${lines[i].trim()}`; i++ }
        else break
      }
      result.push({ kind: 'list', ordered, items })
      continue
    }
    if (!line.trim()) { i++; continue }
    const paragraph: string[] = []
    while (i < lines.length && lines[i].trim() && !/^\s*```/.test(lines[i]) && !/^(#{1,4})\s+/.test(lines[i]) && !/^\s*([-*•]|\d+[.)])\s+/.test(lines[i])) {
      paragraph.push(lines[i++])
    }
    result.push({ kind: 'paragraph', lines: paragraph })
  }
  return result
}

export function AssistantMarkdown({ text }: { text: string }): React.JSX.Element {
  return (
    <div className="space-y-3 text-base leading-relaxed text-ink [overflow-wrap:anywhere]">
      {blocks(text).map((block, index) => {
        const key = `b${index}`
        switch (block.kind) {
          case 'heading':
            return <p key={key} className="pt-1.5 text-base font-semibold text-ink">{inline(block.text, key)}</p>
          case 'code':
            return <pre key={key} className="glass-well overflow-x-auto rounded-xl px-3.5 py-2.5 font-mono text-xs leading-relaxed text-ink">{block.text}</pre>
          case 'list': {
            const Tag = block.ordered ? 'ol' : 'ul'
            return (
              <Tag key={key} className={block.ordered ? 'list-decimal space-y-1.5 pl-5 marker:text-ink-subtle' : 'list-disc space-y-1.5 pl-5 marker:text-ink-subtle'}>
                {block.items.map((item, itemIndex) => <li key={`${key}-${itemIndex}`}>{inline(item, `${key}-${itemIndex}`)}</li>)}
              </Tag>
            )
          }
          default:
            return (
              <p key={key}>
                {block.lines.map((line, lineIndex) => (
                  <Fragment key={`${key}-${lineIndex}`}>{lineIndex > 0 && <br />}{inline(line, `${key}-${lineIndex}`)}</Fragment>
                ))}
              </p>
            )
        }
      })}
    </div>
  )
}
