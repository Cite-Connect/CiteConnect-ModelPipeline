# CiteConnect Frontend Integration Guide

## Overview
CiteConnect provides intelligent paper recommendations using a **single unified API endpoint** that automatically adapts between search-based and profile-based recommendations.

---

## 🎯 API Endpoint

### **Primary Endpoint (Use This)**
```
POST /api/v1/recommendations
```

**Base URL:** `http://localhost:8000` (development)

---

## 📝 Request Format

### **Request Body**
```json
{
  "user_id": 2,
  "count": 10,
  "model_preference": "minilm",
  "search_query": "neural networks for brain segmentation",
  "session_id": "session-abc-123"
}
```

### **Field Descriptions**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | integer | ✅ Yes | Current logged-in user ID |
| `count` | integer | No (default: 10) | Number of papers to return (1-50) |
| `model_preference` | string | No (default: "minilm") | Embedding model: `"minilm"` or `"specter"` |
| `search_query` | string | No (default: null) | Search query text (or null for profile-based) |
| `session_id` | string | ✅ Yes | Unique session identifier |

### **Critical: search_query Field**

```javascript
// User typed something
search_query: "neural networks for brain segmentation"  → Backend uses search-augmented strategy

// User left search box empty
search_query: null  → Backend uses profile-based strategy

// Convert empty strings to null
const query = searchBox.value.trim() || null;
```

---

## 📊 Response Format

### **Response Structure**
```json
{
  "recommendations": [
    {
      "paper_id": "abc123...",
      "title": "Brain Tumor Segmentation Using Neural Networks...",
      "authors": ["Smith, J.", "Doe, A."],
      "year": 2023,
      "citation_count": 450,
      "domain": "healthcare",
      "abstract": "We propose a novel approach...",
      "relevance_score": 0.88,
      "relevance_explanation": "Title matches: neural, networks, brain; Semantically similar to your search",
      "match_source": "keyword+semantic",
      "score_breakdown": {
        "keyword": 0.92,
        "semantic": 0.78,
        "profile": 0.15
      }
    }
  ],
  "metadata": {
    "strategy_used": "search_augmented",
    "search_query": "neural networks for brain segmentation",
    "model_used": "all-MiniLM-L6-v2",
    "generation_time_ms": 850,
    "evaluation_score": 0.76
  }
}
```

### **Key Response Fields**

**Per Paper:**
- `relevance_score` (0.0-1.0): Overall relevance (display as percentage)
- `relevance_explanation`: Why this paper is relevant (show to user)
- `match_source`: How paper was found (`"keyword"`, `"semantic"`, `"profile"`, or combinations)
- `score_breakdown`: Detailed scoring (optional - for advanced UI)

**Metadata:**
- `strategy_used`: Which strategy backend used (`"search_augmented"`, `"cold_start"`, or `"warm_start"`)
- `search_query`: Echo of what user searched (or null)
- `generation_time_ms`: Performance metric

---

## 💻 Implementation Code

### **Minimal Implementation**
```javascript
async function getRecommendations() {
  const searchQuery = document.getElementById('searchBox').value.trim();
  
  const response = await fetch('http://localhost:8000/api/v1/recommendations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: getCurrentUserId(),           // From auth
      count: 10,
      model_preference: 'minilm',
      search_query: searchQuery || null,    // Key line
      session_id: generateSessionId()
    })
  });
  
  const data = await response.json();
  displayResults(data);
}

function displayResults(data) {
  const { recommendations, metadata } = data;
  
  recommendations.forEach((paper, index) => {
    console.log(`${index + 1}. ${paper.title}`);
    console.log(`   Relevance: ${(paper.relevance_score * 100).toFixed(0)}%`);
    console.log(`   Why: ${paper.relevance_explanation}`);
    console.log(`   Match: ${paper.match_source}`);
  });
}
```

### **React/Next.js Implementation**
```jsx
import { useState } from 'react';

function SearchRecommendations() {
  const [searchQuery, setSearchQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    setLoading(true);
    
    try {
      const response = await fetch('/api/v1/recommendations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          count: 10,
          model_preference: 'minilm',
          search_query: searchQuery.trim() || null,
          session_id: `session-${Date.now()}`
        })
      });
      
      const data = await response.json();
      setResults(data);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
        placeholder="Search papers or get recommendations..."
      />
      <button onClick={handleSearch}>Search</button>
      
      {results && (
        <div>
          <p>Strategy: {results.metadata.strategy_used}</p>
          {results.recommendations.map(paper => (
            <div key={paper.paper_id}>
              <h3>{paper.title}</h3>
              <p>⭐ {(paper.relevance_score * 100).toFixed(0)}%</p>
              <p>{paper.relevance_explanation}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

---

## 🔄 How Backend Auto-Routing Works

```
┌──────────────────────────────────────────┐
│  Frontend Sends                          │
│  search_query: "neural networks..."     │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│  Backend Checks                          │
│  if (search_query exists and not empty) │
└────────────┬─────────────────────────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
┌──────────┐  ┌──────────┐
│   YES    │  │    NO    │
│ (has     │  │ (empty/  │
│  query)  │  │  null)   │
└────┬─────┘  └─────┬────┘
     │              │
     ▼              ▼
┌─────────────┐ ┌────────────┐
│ SEARCH MODE │ │ PROFILE    │
│             │ │ MODE       │
│ Strategy:   │ │            │
│ "search_    │ │ Strategy:  │
│ augmented"  │ │ "cold_     │
│             │ │ start"     │
│ Sources:    │ │            │
│ • Keyword   │ │ Sources:   │
│ • Semantic  │ │ • Profile  │
│ • Profile   │ │ • Canonical│
│             │ │ • Ground   │
│ Weights:    │ │   Truth    │
│ 50-40-10    │ │            │
│             │ │ Weights:   │
│             │ │ 40-20-15   │
└─────────────┘ └────────────┘
```

**Frontend doesn't need to know this!** Backend handles routing automatically.

---

## 🎨 UI Display Recommendations

### **Show Strategy Used** (Optional but Helpful)
```javascript
if (metadata.strategy_used === 'search_augmented') {
  showBanner(`Search results for: "${metadata.search_query}"`);
} else {
  showBanner("Personalized recommendations for you");
}
```

### **Display Paper Cards**
```
┌────────────────────────────────────────────┐
│ 🥇 Paper Title Here                        │
│ ⭐ Relevance: 88%                          │
│ 💬 Title matches: neural, brain, networks  │
│ 🏷️ keyword+semantic                       │
│ 👤 Smith et al. | 2023 | 450 citations    │
└────────────────────────────────────────────┘
```

**Minimum fields to display:**
- `title` - Paper title
- `relevance_score` - Show as percentage (88%)
- `relevance_explanation` - Why it's relevant

**Optional enrichment:**
- `match_source` - Badge showing match type
- `score_breakdown` - Visual breakdown bars
- `citation_count` - Paper impact
- `year` - Recency indicator

---

## ⚡ Performance Expectations

| Scenario | Expected Response Time |
|----------|----------------------|
| First search (models loading) | 10-15 seconds |
| Subsequent searches (cached) | 500-1000ms |
| Profile-based (no search) | 400-700ms |

**Loading states:** Always show loading indicator during request.

---

## 🧪 Testing

### **Test Case 1: With Search Query**
```bash
curl -X POST "http://localhost:8000/api/v1/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 2,
    "count": 5,
    "search_query": "neural networks for brain segmentation",
    "session_id": "test-1"
  }'
```

**Expected:** `strategy_used: "search_augmented"`

### **Test Case 2: Without Search Query**
```bash
curl -X POST "http://localhost:8000/api/v1/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 2,
    "count": 5,
    "search_query": null,
    "session_id": "test-2"
  }'
```

**Expected:** `strategy_used: "cold_start"`

---

## ❗ Important Notes

1. **Session ID**: Generate unique ID per session for tracking
   ```javascript
   const sessionId = `session-${userId}-${Date.now()}`;
   ```

2. **Error Handling**: Always handle 500 errors gracefully
   ```javascript
   if (!response.ok) {
     throw new Error(`HTTP ${response.status}: ${response.statusText}`);
   }
   ```

3. **Empty Search**: Convert empty strings to null
   ```javascript
   search_query: searchQuery.trim() || null
   ```

4. **Authentication**: Pass actual user_id from auth context, not hardcoded

---

## 🚀 Quick Start Checklist

- [ ] Single search input box in UI
- [ ] Call `/api/v1/recommendations` on Enter or button click
- [ ] Pass `search_query` from input (or null if empty)
- [ ] Display `recommendations` array
- [ ] Show `relevance_explanation` to explain why each paper is relevant
- [ ] Handle loading state (10s first time, <1s cached)
- [ ] Display strategy used from `metadata.strategy_used`

---

## 📞 Support

**Backend is ready!** No frontend changes needed on backend side.

**Questions?** Check response metadata for debugging:
- `metadata.strategy_used` - Confirms which mode was used
- `metadata.search_query` - Echoes back what was searched
- `metadata.generation_time_ms` - Performance metric