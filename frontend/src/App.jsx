import { useState, useCallback, useRef } from 'react'
import DescriptionInput from './components/DescriptionInput'
import RegexDisplay from './components/RegexDisplay'
import Playground from './components/Playground'
import History from './components/History'

function App() {
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const [loading, setLoading] = useState(false)
    const debounceTimer = useRef(null)

    const hasContent = result || error || loading

    const handleRegexEdit = useCallback((newRegex) => {

        setResult(prev => ({ ...prev, regex: newRegex }))


        if (debounceTimer.current) {
            clearTimeout(debounceTimer.current)
        }


        debounceTimer.current = setTimeout(async () => {
            try {
                const res = await fetch('/api/tokenize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ regex: newRegex }),
                })
                const data = await res.json()
                if (data.tokens) {
                    setResult(prev => ({ ...prev, tokens: data.tokens }))
                }
            } catch (err) {
                console.error('Failed to update tokens:', err)
            }
        }, 300)
    }, [])

    const handleGenerate = useCallback(async (description) => {
        setLoading(true)
        setError(null)
        setResult(null)

        try {
            const res = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ description }),
            })
            const data = await res.json()

            if (data.error) {
                setError(data.error)
            } else {
                setResult(data)
                const history = JSON.parse(localStorage.getItem('regex-history') || '[]')
                const entry = { description, regex: data.regex, timestamp: Date.now() }
                const updated = [entry, ...history.filter(h => h.description !== description)].slice(0, 20)
                localStorage.setItem('regex-history', JSON.stringify(updated))
            }
        } catch (err) {
            setError('Failed to connect to backend. Make sure the Go server is running on :8080')
        } finally {
            setLoading(false)
        }
    }, [])

    const handleHistoryClick = useCallback((entry) => {
        handleGenerate(entry.description)
    }, [handleGenerate])

    return (
        <div className={`max-w-[860px] mx-auto px-6 pb-20 pt-10 flex flex-col ${hasContent ? '' : 'min-h-screen justify-center'}`}>
            <header className="text-center mb-10">
                <h1 className="text-[28px] font-bold text-accent tracking-tight">Regex AI</h1>
                <p className="text-[15px] text-text-secondary mt-1.5">Describe what you need in plain text - get a regex with visual explanation</p>
            </header>

            <DescriptionInput onGenerate={handleGenerate} loading={loading} />

            {loading && (
                <div className="bg-card border border-border rounded-2xl p-6 mb-5 backdrop-blur-sm text-center animate-fade-in">
                    <div className="inline-flex gap-1 items-center">
                        <span className="w-1.5 h-1.5 bg-accent rounded-full dot dot-1"></span>
                        <span className="w-1.5 h-1.5 bg-accent rounded-full dot dot-2"></span>
                        <span className="w-1.5 h-1.5 bg-accent rounded-full dot dot-3"></span>
                    </div>
                </div>
            )}

            {error && (
                <div className="bg-card border border-border rounded-2xl p-6 mb-5 backdrop-blur-sm animate-fade-in">
                    <div className="bg-error/8 border border-error/15 rounded-xl p-3 px-4 text-[13px] text-error">{error}</div>
                </div>
            )}

            {result && (
                <>
                    <RegexDisplay regex={result.regex} tokens={result.tokens} onEdit={handleRegexEdit} />
                    <Playground regex={result.regex} />
                </>
            )}

            <History onSelect={handleHistoryClick} />
        </div>
    )
}

export default App
