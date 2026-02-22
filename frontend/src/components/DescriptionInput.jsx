import { useState } from 'react'

function DescriptionInput({ onGenerate, loading }) {
    const [value, setValue] = useState('')

    const handleSubmit = (e) => {
        e.preventDefault()
        if (value.trim() && !loading) {
            onGenerate(value.trim())
        }
    }

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSubmit(e)
        }
    }

    return (
        <div className="bg-card border border-border rounded-2xl p-6 mb-5 backdrop-blur-sm shadow-xs transition-all hover:border-black/12 hover:shadow-sm">
            <div className="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-4">
                Describe your pattern
            </div>
            <form onSubmit={handleSubmit}>
                <div className="flex gap-3">
                    <input
                        type="text"
                        className="flex-1 bg-input border border-border rounded-xl py-3.5 px-4.5 text-[15px] font-sans text-text outline-none transition-all placeholder:text-text-muted focus:border-border-focus focus:shadow-[0_0_0_3px_rgba(50,50,60,0.06)]"
                        placeholder="e.g. find all email addresses, validate phone UA, match URLs..."
                        value={value}
                        onChange={(e) => setValue(e.target.value)}
                        onKeyDown={handleKeyDown}
                        disabled={loading}
                        autoComplete="off"
                        autoFocus
                    />
                    <button
                        type="submit"
                        className="bg-gradient-to-br from-accent to-accent-light border-none rounded-xl py-3.5 px-7 text-sm font-semibold text-white cursor-pointer transition-all whitespace-nowrap shadow-md hover:-translate-y-px hover:shadow-lg active:translate-y-0 disabled:opacity-40 disabled:cursor-default disabled:translate-y-0"
                        disabled={loading || !value.trim()}
                    >
                        Generate
                    </button>
                </div>
            </form>
        </div>
    )
}

export default DescriptionInput
