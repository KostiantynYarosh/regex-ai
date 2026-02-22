import { useState, useEffect } from 'react'

function RegexDisplay({ regex, tokens, onEdit }) {
    const [copied, setCopied] = useState(false)
    const [localRegex, setLocalRegex] = useState(regex)


    useEffect(() => {
        setLocalRegex(regex)
    }, [regex])

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(localRegex)
            setCopied(true)
            setTimeout(() => setCopied(false), 2000)
        } catch {
            const el = document.createElement('textarea')
            el.value = localRegex
            document.body.appendChild(el)
            el.select()
            document.execCommand('copy')
            document.body.removeChild(el)
            setCopied(true)
            setTimeout(() => setCopied(false), 2000)
        }
    }

    const handleChange = (e) => {
        const val = e.target.value
        setLocalRegex(val)
        onEdit(val)
    }

    return (
        <div className="bg-card border border-border rounded-2xl p-6 mb-5 backdrop-blur-sm shadow-xs transition-all hover:border-black/12 hover:shadow-sm animate-fade-in">
            <div className="flex items-center justify-between mb-4">
                <div className="text-xs font-semibold uppercase tracking-wider text-text-secondary">
                    Generated Regex
                </div>
                <button
                    className="bg-transparent border border-border rounded-md py-1 px-2.5 text-xs text-text-secondary cursor-pointer font-sans transition-all hover:text-text select-none"
                    onClick={handleCopy}
                >
                    {copied ? 'Copied' : 'Copy'}
                </button>
            </div>


            <textarea
                className="w-full bg-input border border-border rounded-xl p-4.5 font-mono text-[17px] leading-relaxed text-text outline-none resize-none transition-all focus:border-border-focus focus:shadow-[0_0_0_3px_rgba(50,50,60,0.06)] mb-5 min-h-[100px] no-scrollbar overflow-hidden"
                value={localRegex}
                onChange={handleChange}
                spellCheck="false"
                rows={3}
                onInput={(e) => {
                    e.target.style.height = 'auto'
                    e.target.style.height = e.target.scrollHeight + 'px'
                }}
            />


            {tokens && tokens.length > 0 && (
                <div className="animate-fade-in select-text">
                    <div className="text-[11px] font-bold uppercase tracking-widest text-text-muted mb-3">Visual Breakdown</div>
                    <div className="block whitespace-pre-wrap break-all leading-[2.2] selection:bg-accent/20 pl-1">
                        {tokens.map((token, i) => (
                            <span
                                key={i}
                                className="group relative inline py-0.5 px-0.5 rounded cursor-pointer transition-all font-medium text-[16.5px] select-text"
                                style={{
                                    color: token.color,
                                    backgroundColor: token.color + '10',
                                }}
                            >
                                {token.value}
                                <span className="pointer-events-none absolute bottom-[calc(100%+8px)] left-1/2 -translate-x-1/2 bg-[#1a1a1a] border border-white/10 rounded-lg py-2 px-3.5 text-xs font-sans font-normal text-white whitespace-nowrap opacity-0 scale-95 group-hover:opacity-100 group-hover:scale-100 transition-all z-10 shadow-xl select-none">
                                    {token.label}
                                    <span className="absolute top-full left-1/2 -translate-x-1/2 border-[5px] border-transparent border-t-[#1a1a1a]"></span>
                                </span>
                            </span>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}

export default RegexDisplay
