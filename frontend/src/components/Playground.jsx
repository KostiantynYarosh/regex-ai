import { useState, useEffect, useCallback } from 'react'

function Playground({ regex }) {
    const [testText, setTestText] = useState('')
    const [matches, setMatches] = useState([])
    const [error, setError] = useState(null)

    const validate = useCallback(async () => {
        if (!testText.trim() || !regex) {
            setMatches([])
            setError(null)
            return
        }

        try {
            const res = await fetch('/api/validate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ regex, testText }),
            })
            const data = await res.json()

            if (data.error) {
                setError(data.error)
                setMatches([])
            } else {
                setError(null)
                setMatches(data.matches || [])
            }
        } catch {
            setError('Failed to validate - check backend connection')
        }
    }, [regex, testText])

    useEffect(() => {
        const timer = setTimeout(validate, 300)
        return () => clearTimeout(timer)
    }, [validate])

    const renderHighlighted = () => {
        if (!testText || matches.length === 0) return testText

        const parts = []
        let lastIndex = 0

        const sorted = [...matches].sort((a, b) => a.start - b.start)
        for (const m of sorted) {
            if (m.start > lastIndex) {
                parts.push(<span key={`t-${lastIndex}`}>{testText.slice(lastIndex, m.start)}</span>)
            }
            parts.push(
                <span key={`m-${m.start}`} className="match-hl">{testText.slice(m.start, m.end)}</span>
            )
            lastIndex = m.end
        }
        if (lastIndex < testText.length) {
            parts.push(<span key={`t-${lastIndex}`}>{testText.slice(lastIndex)}</span>)
        }
        return parts
    }

    return (
        <div className="bg-card border border-border rounded-2xl p-6 mb-5 backdrop-blur-sm shadow-xs transition-all hover:border-black/12 hover:shadow-sm animate-fade-in">
            <div className="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-4">
                Live Playground
            </div>

            <textarea
                className="w-full min-h-[100px] bg-input border border-border rounded-xl py-3.5 px-4.5 text-[15px] font-mono text-text outline-none resize-none transition-all leading-relaxed placeholder:text-text-muted placeholder:font-sans focus:border-border-focus focus:shadow-[0_0_0_3px_rgba(50,50,60,0.06)] no-scrollbar overflow-hidden"
                placeholder="Paste your test text here to see matches in real-time..."
                value={testText}
                onChange={(e) => setTestText(e.target.value)}
                onInput={(e) => {
                    e.target.style.height = 'auto'
                    e.target.style.height = e.target.scrollHeight + 'px'
                }}
            />

            {error && (
                <div className="bg-error/8 border border-error/15 rounded-xl p-3 px-4 text-[13px] text-error mt-3">{error}</div>
            )}

            {testText && matches.length > 0 && (
                <div className="mt-4 animate-fade-in">
                    <div className="text-xs font-semibold uppercase tracking-wide text-text-secondary mb-2.5">Highlighted Matches</div>
                    <div className="font-mono text-sm leading-relaxed bg-input rounded-xl py-3.5 px-4.5 border border-border whitespace-pre-wrap break-all">
                        {renderHighlighted()}
                    </div>
                    <div className="inline-flex items-center gap-1.5 text-xs text-success mt-2.5 font-medium">
                        {matches.length} match{matches.length !== 1 ? 'es' : ''} found
                    </div>
                </div>
            )}

            {testText && !error && matches.length === 0 && (
                <div className="text-xs text-text-muted mt-3 font-medium">
                    No matches found
                </div>
            )}
        </div>
    )
}

export default Playground
