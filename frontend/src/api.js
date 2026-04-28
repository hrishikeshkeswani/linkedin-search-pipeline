const BASE = import.meta.env.VITE_API_URL ?? '/api'

export async function search({ query, k = 25, expand = true, rerank = true, synthesize = false, filters = null, sortBy = 'relevance', postType = 'all' }) {
  const res = await fetch(`${BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, k, expand_query: expand, rerank, synthesize, filters, sort_by: sortBy, post_type: postType }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail ?? `Request failed: ${res.status}`)
  }
  return res.json()
}

export async function health() {
  const res = await fetch(`${BASE}/health`)
  return res.json()
}
