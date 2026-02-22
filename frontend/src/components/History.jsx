import { useState, useEffect } from 'react'

function History({ onSelect }) {
    const [history, setHistory] = useState([])

    useEffect(() => {
        const stored = JSON.parse(localStorage.getItem('regex-history') || '[]')
        setHistory(stored)
    }, [])

    useEffect(() => {
        const handleFocus = () => {
            const stored = JSON.parse(localStorage.getItem('regex-history') || '[]')
            setHistory(stored)
        }
        window.addEventListener('focus', handleFocus)
        return () => window.removeEventListener('focus', handleFocus)
    }, [])

    const refreshHistory = () => {
        const stored = JSON.parse(localStorage.getItem('regex-history') || '[]')
        setHistory(stored)
    }

    const handleDelete = (e, index) => {
        e.stopPropagation()
        const updated = history.filter((_, i) => i !== index)
        localStorage.setItem('regex-history', JSON.stringify(updated))
        setHistory(updated)
    }

    const handleSelect = (entry) => {
        onSelect(entry)
        setTimeout(refreshHistory, 500)
    }

    if (history.length === 0) return null

    return (
        <div className="bg-card border border-border rounded-2xl p-6 mb-5 backdrop-blur-sm shadow-xs transition-all hover:border-black/12 hover:shadow-sm">
            <div className="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-4">
                History
            </div>
            <ul className="list-none flex flex-col gap-1.5">
                {history.map((entry, i) => (
                    <li
                        key={i}
                        className="flex items-center justify-between py-2.5 px-3.5 bg-input border border-transparent rounded-lg cursor-pointer transition-all gap-3 hover:border-border hover:bg-[#ededf0]"
                        onClick={() => handleSelect(entry)}
                    >
                        <span className="text-[13px] text-text flex-1 overflow-hidden text-ellipsis whitespace-nowrap">{entry.description}</span>
                        <span className="text-[11px] text-text-muted font-mono max-w-[200px] overflow-hidden text-ellipsis whitespace-nowrap">{entry.regex}</span>
                        <button
                            className="bg-transparent border-none text-text-muted cursor-pointer text-sm p-0.5 transition-colors leading-none hover:text-error"
                            onClick={(e) => handleDelete(e, i)}
                            title="Remove from history"
                        >
                            ✕
                        </button>
                    </li>
                ))}
            </ul>
        </div>
    )
}

export default History
