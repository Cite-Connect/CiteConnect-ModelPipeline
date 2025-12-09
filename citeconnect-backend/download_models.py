# download_models.py
import os
import sys
from sentence_transformers import SentenceTransformer

print('🚀 Downloading embedding models...')
cache_dir = '/app/models'
os.makedirs(cache_dir, exist_ok=True)

# Model 1: MiniLM
try:
    print('📥 Downloading all-MiniLM-L6-v2...')
    model1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=cache_dir)
    print('✅ all-MiniLM-L6-v2 downloaded successfully')
    test_embedding = model1.encode('test', show_progress_bar=False)
    print(f'✅ MiniLM test passed - shape: {test_embedding.shape}')
except Exception as e:
    print(f'❌ MiniLM download failed: {e}')
    sys.exit(1)

# Model 2: SPECTER
try:
    print('📥 Downloading allenai/specter2_base...')
    model2 = SentenceTransformer('allenai/specter', cache_folder=cache_dir)
    print('✅ specter2_base downloaded successfully')
    test_embedding2 = model2.encode('test', show_progress_bar=False)
    print(f'✅ SPECTER test passed - shape: {test_embedding2.shape}')
except Exception as e:
    print(f'⚠️ SPECTER download failed (continuing): {e}')

print('🎉 Model downloads complete!')