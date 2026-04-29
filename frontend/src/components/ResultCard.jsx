export default function ResultCard({ result, rank }) {
  const { author, author_title, text, likes, posted_at, roles, skills, is_hiring, url, score, llm_score } = result

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-5 flex flex-col gap-3 hover:border-slate-600 transition-colors">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <div className="h-9 w-9 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-sm shrink-0">
            {author?.[0]?.toUpperCase() ?? '?'}
          </div>
          <div className="min-w-0">
            <p className="font-semibold text-slate-100 truncate">{author ?? 'Unknown'}</p>
            {author_title && <p className="text-xs text-slate-400 truncate">{author_title}</p>}
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {is_hiring && (
            <span className="px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 text-xs font-medium border border-emerald-500/30">
              Hiring
            </span>
          )}
          <span className="text-xs text-slate-500">#{rank}</span>
        </div>
      </div>

      {/* Post text */}
      <p className="text-sm text-slate-300 leading-relaxed line-clamp-4">{text}</p>

      {/* Skills + Roles */}
      {(skills?.length > 0 || roles?.length > 0) && (
        <div className="flex flex-wrap gap-1.5">
          {roles?.slice(0, 3).map(r => (
            <span key={r} className="px-2 py-0.5 rounded-md bg-purple-500/15 text-purple-300 text-xs border border-purple-500/20">{r}</span>
          ))}
          {skills?.slice(0, 5).map(s => (
            <span key={s} className="px-2 py-0.5 rounded-md bg-blue-500/15 text-blue-300 text-xs border border-blue-500/20">{s}</span>
          ))}
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between text-xs text-slate-500 pt-1 border-t border-slate-700">
        <div className="flex items-center gap-3">
          {likes > 0 && <span>♥ {likes.toLocaleString()}</span>}
          {posted_at && <span>{new Date(posted_at).toLocaleDateString()}</span>}
        </div>
        <div className="flex items-center gap-2">
          <span>score {score.toFixed(3)}</span>
          {url && (
            <a href={url} target="_blank" rel="noopener noreferrer"
              className="text-blue-400 hover:text-blue-300 underline underline-offset-2">
              View
            </a>
          )}
        </div>
      </div>
    </div>
  )
}
