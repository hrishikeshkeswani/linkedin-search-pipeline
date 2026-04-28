import { useState } from 'react'

export default function SearchBar({ onSearch, loading }) {
  const [query, setQuery] = useState('')
  const [sortBy, setSortBy] = useState('relevance')
  const [k, setK] = useState(25)
  const [postType, setPostType] = useState('all')

  function handleSubmit(e) {
    e.preventDefault()
    if (!query.trim()) return
    onSearch({ query: query.trim(), sortBy, k, postType, synthesize: false })
  }

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="e.g. senior ML engineer remote, Python LangChain…"
          className="flex-1 px-4 py-3 rounded-xl bg-slate-800 border border-slate-700 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 text-base"
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-medium transition-colors"
        >
          {loading ? (
            <span className="flex items-center gap-2"><Spinner /> Searching</span>
          ) : 'Search'}
        </button>
      </div>

      <div className="flex items-center gap-4 mt-3 flex-wrap">
        <div className="flex items-center gap-1">
          <span className="text-sm text-slate-500 mr-2">Sort by</span>
          {['relevance', 'date'].map(opt => (
            <button
              key={opt}
              type="button"
              onClick={() => setSortBy(opt)}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors capitalize ${
                sortBy === opt
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              {opt}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1">
          <span className="text-sm text-slate-500 mr-2">Show</span>
          {[10, 25, 50].map(n => (
            <button
              key={n}
              type="button"
              onClick={() => setK(n)}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                k === n
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              {n}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1">
          <span className="text-sm text-slate-500 mr-2">Type</span>
          {[['all', 'All'], ['hiring', 'Hiring Posts'], ['jobs', 'Job Posts']].map(([val, label]) => (
            <button
              key={val}
              type="button"
              onClick={() => setPostType(val)}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                postType === val
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    </form>
  )
}

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
    </svg>
  )
}
