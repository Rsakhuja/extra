---
title: Cross-Request LRU Caching
impact: HIGH
impactDescription: caches across requests
tags: server, cache, lru, cross-request
---

## Cross-Request LRU Caching

`React.cache()` only works within one request. For data shared across sequential requests (user clicks button A then button B), use an LRU cache.

**Implementation:**

```typescript
import { LRUCache } from 'lru-cache'

const cache = new LRUCache<string, any>({
  max: 1000,
  ttl: 5 * 60 * 1000  // 5 minutes
})

// Key by every input that changes the result, never by the record ID alone
export async function getUser(tenantId: string, id: string) {
  const key = `${tenantId}:${id}`
  const cached = cache.get(key)
  if (cached) return cached

  const user = await db.user.findFirst({ where: { tenantId, id } })
  cache.set(key, user)
  return user
}

// Request 1: DB query, result cached
// Request 2: cache hit, no DB query
```

Use when sequential user actions hit multiple endpoints needing the same data within seconds.

This cache is shared by every request in the process. Include the tenant and any other scope that changes the result in the key, and check the caller's authorization on every request before returning cached data; never cache the authorization decision or caller-specific views. Keep the TTL short, because permission and data changes take effect only when an entry expires.

**With Vercel's [Fluid Compute](https://vercel.com/docs/fluid-compute):** LRU caching is especially effective because multiple concurrent requests can share the same function instance and cache. This means the cache persists across requests without needing external storage like Redis.

**In traditional serverless:** Each invocation runs in isolation, so consider Redis for cross-process caching.

Reference: [https://github.com/isaacs/node-lru-cache](https://github.com/isaacs/node-lru-cache)
