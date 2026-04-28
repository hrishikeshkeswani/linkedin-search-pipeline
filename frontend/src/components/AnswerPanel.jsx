export default function AnswerPanel({ answer }) {
  if (!answer) return null
  return (
    <div className="w-full rounded-xl border border-blue-500/30 bg-blue-950/30 p-5 text-sm leading-relaxed text-slate-200">
      <div className="flex items-center gap-2 mb-3 text-blue-400 font-medium text-xs uppercase tracking-wider">
        <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
        AI Summary
      </div>
      <p className="whitespace-pre-wrap">{answer}</p>
    </div>
  )
}
