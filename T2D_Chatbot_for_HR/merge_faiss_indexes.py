import os
import faiss
import pickle
import numpy as np
from pathlib import Path

# Dossier contenant les index par profil
VECTORSTORES_DIR = Path("backend/rag_service/vectorstores_faiss")
PROFILES = [d for d in VECTORSTORES_DIR.iterdir() if d.is_dir()]

all_embeddings = []
all_texts = []
all_metadatas = []
index = None

for profile_dir in PROFILES:
    index_path = profile_dir / "index.faiss"
    texts_path = profile_dir / "texts.pkl"
    metadata_path = profile_dir / "metadata.pkl"
    if not index_path.exists():
        continue
    print(f"Chargement de {index_path} ...")
    idx = faiss.read_index(str(index_path))
    with open(texts_path, "rb") as f:
        texts = pickle.load(f)
    with open(metadata_path, "rb") as f:
        metadatas = pickle.load(f)
    # Récupérer les embeddings
    xb = faiss.vector_to_array(idx.reconstruct_n(
        0, idx.ntotal)).reshape(idx.ntotal, -1)
    all_embeddings.append(xb)
    all_texts.extend(texts)
    all_metadatas.extend(metadatas)
    if index is None:
        # Créer un nouvel index avec la même dimension
        index = faiss.IndexFlatL2(xb.shape[1])

if not all_embeddings:
    print("Aucun index trouvé.")
    exit(1)

all_embeddings = np.vstack(all_embeddings)
index.add(all_embeddings.astype('float32'))

faiss.write_index(index, "faiss.index")
with open("faiss_texts.pkl", "wb") as f:
    pickle.dump(all_texts, f)
with open("faiss_metadatas.pkl", "wb") as f:
    pickle.dump(all_metadatas, f)

print(f"Index global créé avec {index.ntotal} vecteurs.")
