export default function StatusBar({ nResults, query, expanded, reranked }) {
  if (!query) return null
  return (
    <div className="flex items-center gap-4 text-xs text-slate-500">
      <span>
        <span className="text-slate-300 font-medium">{nResults}</span> results for "
        <span className="text-slate-300">{query}</span>"
      </span>
      {expanded && <Badge label="Query expanded" color="blue" />}
      {reranked && <Badge label="LLM reranked" color="purple" />}
    </div>
  )
}

function Badge({ label, color }) {
  const colors = {
    blue: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    purple: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
  }
  return (
    <span className={`px-2 py-0.5 rounded-full border text-xs ${colors[color]}`}>{label}</span>
  )
}
