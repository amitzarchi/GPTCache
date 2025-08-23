import time
import os

import sys
# Add the path to your cloned GPTCache repo
sys.path.insert(0, r'C:\Users\yossi_1wprdhx\Desktop\llmcache\GPTCache')  # use the GPTCache repo and not the package

def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']


print("🧪 Quality Score Eviction Test - Cache Eviction Behavior")
print("=" * 60)

# Enhanced GPTCache setup with Quality Score Eviction
# -------------------------------------------------
from gptcache import cache
from gptcache.adapter import openai
from gptcache.manager import manager_factory
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation import SearchDistanceEvaluation

# Clean up any existing cache data
if os.path.exists("./cache_data"):
    import shutil

    shutil.rmtree("./cache_data")
    print("🧹 Cleaned up existing cache data")

# Create data manager with SMALL cache for eviction testing
data_manager = manager_factory(
    manager="sqlite,faiss",
    data_dir="./cache_data",
    eviction_manager="quality_score",
    vector_params={"dimension": 768},
    eviction_params={
        "maxsize": 4,  # SMALL cache - eviction will happen!
        "clean_size": 1,  # Evict 1 item at a time for clear demonstration
        "learning_rate": 0.3,  # Learn reasonably fast
        "quality_weight": 0.8,  # Heavily prioritize quality
        "recency_weight": 0.15,  # Some recency consideration
        "frequency_weight": 0.05  # Minimal frequency impact
    }
)

cache.init(
    data_manager=data_manager,
    embedding_func=Onnx().to_embeddings,
    similarity_evaluation=SearchDistanceEvaluation()
)

os.environ['OPENAI_API_KEY'] = 'mockup'
cache.set_openai_key()
print("✅ Cache initialized with Quality Score Eviction (maxsize=4)")


def show_cache_stats(step_name):
    """Display current cache statistics"""
    if hasattr(data_manager, 'eviction_base') and hasattr(data_manager.eviction_base, 'get_stats'):
        stats = data_manager.eviction_base.get_stats()
        print(f"\n📊 {step_name}:")
        print(f"   Cache: {stats['size']}/{stats['max_size']} items")
        print(f"   Avg Quality: {stats['avg_quality']:.3f}")
        print(f"   Quality Range: {stats.get('min_quality', 0):.3f} - {stats.get('max_quality', 0):.3f}")

        # Show individual item qualities if we can access them
        if hasattr(data_manager.eviction_base, 'entries'):
            print("   Individual Items:")
            for obj_id, entry in data_manager.eviction_base.entries.items():
                composite_score = data_manager.eviction_base._calculate_composite_score(entry)
                print(
                    f"     Item {obj_id}: quality={entry.quality_score:.3f}, composite={composite_score:.3f}, access_count={entry.access_count}")
        print()


# Test questions designed to create different quality patterns
test_questions = [
    # Phase 1: Fill the cache (4 items)
    ("What is the capital of France?", "france_capital"),
    ("What is the weather today?", "weather"),
    ("Explain machine learning", "ml_explain"),
    ("What is 2+2?", "math_simple"),

    # Phase 2: Trigger evictions (these will cause low-quality items to be evicted)
    ("What's the capital city of France?", "france_capital_similar"),  # High similarity to first question
    ("Tell me about Python programming", "python_prog"),  # New topic
    ("What is the capital of France again?", "france_capital_repeat"),  # High similarity again
    ("How does quantum computing work?", "quantum"),  # New topic
]

print("\n🎯 EVICTION TEST PHASES:")
print("Phase 1: Fill cache with 4 diverse questions")
print("Phase 2: Add more questions to trigger eviction of low-quality items")
print("Phase 3: Test evicted items to verify they were removed")
print("=" * 60)

# Track which items were initially in cache for eviction verification
initial_questions = {}

for i, (question, topic_id) in enumerate(test_questions, 1):
    phase = "Phase 1" if i <= 4 else "Phase 2"
    print(f"\n🔸 {phase} - Question {i}: {question}")

    # Track initial questions for later eviction testing
    if i <= 4:
        initial_questions[i] = (question, topic_id)

    start_time = time.time()
    response = openai.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': question}],
    )

    elapsed_time = time.time() - start_time
    answer = response_text(response)

    # Determine if this was a cache hit or miss based on response time
    cache_status = "🟢 CACHE HIT" if elapsed_time < 0.5 else "🔴 CACHE MISS"
    print(f"   {cache_status} - Time: {elapsed_time:.2f}s")
    print(f"   Answer: {answer[:80]}{'...' if len(answer) > 80 else ''}")

    show_cache_stats(f"After Question {i}")

    # Add small delay between questions
    if i < len(test_questions):
        time.sleep(0.3)

    # Add separator between phases
    if i == 4:
        print("\n" + "🚨 CACHE IS FULL! Next questions will trigger evictions!" + "\n")
        print("=" * 60)

# Phase 3: Test evicted items
print("\n🔍 PHASE 3: EVICTION VERIFICATION")
print("Testing original questions to see which were evicted...")
print("=" * 60)

# Get current cache state
current_items = set()
if hasattr(data_manager.eviction_base, 'entries'):
    current_items = set(data_manager.eviction_base.entries.keys())

# Test each original question to see if it's still cached
evicted_items = []
survived_items = []

for item_id, (question, topic_id) in initial_questions.items():
    print(f"\n🔸 Phase 3 - Testing Original Question {item_id}: {question}")

    start_time = time.time()
    response = openai.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': question}],
    )

    elapsed_time = time.time() - start_time
    answer = response_text(response)

    if elapsed_time < 0.5:
        cache_status = "🟢 STILL CACHED"
        survived_items.append((item_id, topic_id, question))
    else:
        cache_status = "🔴 EVICTED"
        evicted_items.append((item_id, topic_id, question))

    print(f"   {cache_status} - Time: {elapsed_time:.2f}s")
    print(f"   Answer: {answer[:80]}{'...' if len(answer) > 80 else ''}")

    time.sleep(0.3)

print("\n🎯 EVICTION TEST COMPLETE!")
print("=" * 60)

# Final analysis
show_cache_stats("FINAL STATE")

print("🔍 EVICTION ANALYSIS:")
print(f"📤 EVICTED ITEMS ({len(evicted_items)}):")
for item_id, topic_id, question in evicted_items:
    print(f"   • Item {item_id} ({topic_id}): {question}")

print(f"\n💾 SURVIVED ITEMS ({len(survived_items)}):")
for item_id, topic_id, question in survived_items:
    print(f"   • Item {item_id} ({topic_id}): {question}")

print("\n✅ Expected Behavior:")
print("   • Phase 1: Cache fills up with 4 different questions")
print("   • Phase 2: Similar questions (France capital) get high quality scores")
print("   • Low-quality/unrelated items get evicted to make room")
print("   • Phase 3: Evicted items show CACHE MISS, surviving items show CACHE HIT")

print("\n💡 Key Observations:")
print("   • High-quality items (France capital) should survive multiple evictions")
print("   • Low-quality/unrelated items should be evicted first")
print("   • Evicted items take longer to respond (> 1.0s) when queried again")
print("   • Quality Score Eviction successfully prioritizes content relevance!")

# Show which items were actually evicted vs survived
if hasattr(data_manager.eviction_base, 'entries'):
    final_items = list(data_manager.eviction_base.entries.keys())
    print(f"\n🔢 Cache Item IDs Evolution:")
    print(f"   Initial items: [1, 2, 3, 4]")
    print(f"   Final items: {final_items}")
    print(f"   Items evicted during test: {[i for i in [1, 2, 3, 4] if i not in final_items]}")
    print(f"   New items added: {[i for i in final_items if i not in [1, 2, 3, 4]]}")