# Quality Score Eviction Policy

The Quality Score Eviction Policy is a new, intelligent eviction strategy for GPTCache that prioritizes keeping high-quality cache entries based on their similarity scores and access patterns.

## Overview

Traditional eviction policies like LRU, LFU, and FIFO make eviction decisions based solely on access patterns (time or frequency), without considering the **quality** or **relevance** of cached content. The Quality Score Eviction Policy addresses this limitation by:

1. **Tracking Similarity Scores**: Monitors how well cached entries match user queries over time
2. **Learning Quality Patterns**: Builds quality scores based on actual similarity evaluation results
3. **Multi-Factor Decisions**: Considers quality, recency, and frequency when making eviction decisions
4. **Adaptive Behavior**: Automatically adjusts to changing usage patterns

## How It Works

### Quality Measurement

Quality is measured by tracking similarity evaluation scores over time:

```python
# When a user asks: "What is Python?"
# And we have cached: "What is Python programming?" -> "Python is a language..."
# Similarity score might be 0.92

# Quality score is updated using exponential moving average:
new_quality = (1 - learning_rate) * old_quality + learning_rate * similarity_score
```

### Composite Scoring

The eviction decision combines multiple factors:

- **Quality Component** (60% default): Based on similarity scores and consistency
- **Recency Component** (30% default): How recently the item was accessed
- **Frequency Component** (10% default): How often the item is accessed

### Eviction Process

Items with the lowest composite scores are evicted first, ensuring that:
- High-quality, frequently accessed items stay longer
- Low-quality items are evicted quickly
- Recent activity is considered but doesn't override quality

