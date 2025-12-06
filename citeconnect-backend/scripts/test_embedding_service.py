"""Test the new embedding service."""
from app.services.bootstrap.embedding_service import EmbeddingService

# Initialize service (loads models)
print("Initializing EmbeddingService...")
service = EmbeddingService()

# Test 1: Health check
print("\n1️⃣ Health Check:")
health = service.health_check()
for model, status in health.items():
    print(f"   {model}: {status}")

# Test 2: Encode single text with MiniLM
print("\n2️⃣ MiniLM Encoding:")
text = "machine learning for medical imaging"
embedding = service.encode_text(text, model='minilm')
print(f"   Text: {text}")
print(f"   Embedding shape: {embedding.shape}")
print(f"   Sample values: [{embedding[0]:.3f}, {embedding[1]:.3f}, {embedding[2]:.3f}, ...]")

# Test 3: Encode same text with SPECTER
print("\n3️⃣ SPECTER Encoding:")
embedding_specter = service.encode_text(text, model='specter')
print(f"   Text: {text}")
print(f"   Embedding shape: {embedding_specter.shape}")
print(f"   Sample values: [{embedding_specter[0]:.3f}, {embedding_specter[1]:.3f}, {embedding_specter[2]:.3f}, ...]")

# Test 4: Batch encoding
print("\n4️⃣ Batch Encoding:")
texts = [
    "machine learning",
    "medical imaging",
    "diagnostics"
]
batch_embeddings = service.encode_batch(texts, model='minilm')
print(f"   Texts: {len(texts)}")
print(f"   Embeddings shape: {batch_embeddings.shape}")

# Test 5: Model info
print("\n5️⃣ Model Information:")
for model in ['minilm', 'specter']:
    info = service.get_model_info(model)
    print(f"\n   {model.upper()}:")
    print(f"      Loaded: {info['loaded']}")
    print(f"      Dimensions: {info['dimensions']}")
    print(f"      Device: {info['device']}")

print("\n✅ All tests complete!")