import { useState } from 'react'
import { search } from './api'
import SearchBar from './components/SearchBar'
import AnswerPanel from './components/AnswerPanel'
import ResultCard from './components/ResultCard'
import StatusBar from './components/StatusBar'

export default function App() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [data, setData] = useState(null)
  const [lastQuery, setLastQuery] = useState('')

  async function handleSearch(params) {
    setLoading(true)
    setError(null)
    setLastQuery(params.query)
    try {
      const result = await search(params)
      setData(result)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center gap-3">
          <div className="h-8 w-8 rounded-lg bg-blue-600 flex items-center justify-center">
            <svg className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <div>
            <h1 className="text-base font-semibold text-white leading-none">LinkedIn Search</h1>
            <p className="text-xs text-slate-500">RAG-powered job search</p>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-10 flex flex-col gap-8">
        {/* Hero */}
        {!data && !loading && (
          <div className="text-center py-8">
            <h2 className="text-4xl font-bold text-white mb-3 tracking-tight">
              Find your next role
            </h2>
            <p className="text-slate-400 text-lg max-w-xl mx-auto">
              Natural language search over 100K+ LinkedIn posts, powered by FAISS + Groq.
            </p>
          </div>
        )}

        {/* Search */}
        <SearchBar onSearch={handleSearch} loading={loading} />

        {/* Error */}
        {error && (
          <div className="rounded-xl border border-red-500/30 bg-red-950/20 px-4 py-3 text-red-400 text-sm">
            {error}
          </div>
        )}

        {/* Loading skeleton */}
        {loading && (
          <div className="flex flex-col gap-3">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-40 rounded-xl bg-slate-800 animate-pulse" />
            ))}
          </div>
        )}

        {/* Results */}
        {data && !loading && (
          <div className="flex flex-col gap-5">
            <StatusBar
              nResults={data.n_results}
              query={lastQuery}
              expanded={data.expanded_queries_used}
              reranked={data.reranked}
            />
            <AnswerPanel answer={data.answer} />
            <div className="flex flex-col gap-3">
              {data.results.map((r, i) => (
                <ResultCard key={r.post_id} result={r} rank={i + 1} />
              ))}
            </div>
            {data.n_results === 0 && (
              <div className="text-center py-12 text-slate-500">
                No results found. Try a different query.
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
