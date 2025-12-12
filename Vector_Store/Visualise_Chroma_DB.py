"""
visualize_chroma_graph.py
- Edit: PERSIST_DIR, COLLECTION_NAME, K, USE_UMAP
- Output: PNG saved as chroma_graph.png and CSV saved as chroma_graph_nodes.csv
"""
import os
import sys
from chromadb.config import Settings
import chromadb
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# ---------- CONFIG (edit these) ----------
PERSIST_DIR = ".Vector_Store/chroma_db"   # path to your chroma persist directory
COLLECTION_NAME = "rag_collection"        # name of your collection
K = 5                                     # neighbors per node for kNN graph
USE_UMAP = False                          # set True to use UMAP instead of PCA (better for clusters)
UMAP_N_NEIGHBORS = 15
SHOW_LABELS_THRESHOLD = 100               # show node labels only if n_nodes <= this
OUTPUT_PNG = "chroma_graph.png"
OUTPUT_CSV = "chroma_graph_nodes.csv"
# -----------------------------------------

def load_collection(persist_dir, collection_name):
    # Use PersistentClient for disk-backed Chroma instances
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        col = client.get_collection(collection_name)
    except Exception as e:
        raise RuntimeError(f"Could not open collection '{collection_name}' at '{persist_dir}': {e}")
    return col

def fetch_embeddings_and_meta(collection):
    # include embeddings can be heavy; but required for visualization
    res = collection.get(include=["embeddings", "ids", "documents", "metadatas"])
    embeddings = np.array(res.get("embeddings", []))
    ids = res.get("ids", [])
    docs = res.get("documents", [None]*len(ids))
    metas = res.get("metadatas", [None]*len(ids))
    if embeddings.size == 0:
        raise RuntimeError("No embeddings found in collection (embeddings array is empty).")
    return embeddings, ids, docs, metas

def normalize_embeddings(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return emb / norms

def build_knn_graph(emb_norm, ids, k):
    n = emb_norm.shape[0]
    k_search = min(k+1, n)  # +1 because nearest includes self
    nbrs = NearestNeighbors(n_neighbors=k_search, metric="cosine", algorithm="auto").fit(emb_norm)
    distances, indices = nbrs.kneighbors(emb_norm)
    G = nx.Graph()
    for i, node_id in enumerate(ids):
        G.add_node(node_id)
    for i, neigh_idxs in enumerate(indices):
        src = ids[i]
        for idx_pos, j in enumerate(neigh_idxs):
            if j == i:
                continue
            tgt = ids[j]
            # distance is 1 - cosine_similarity; convert to similarity in [0,1]
            dist = float(distances[i, idx_pos])
            sim = max(0.0, 1.0 - dist)
            # add edge (avoid duplicate overhead; networkx ignores duplicates)
            G.add_edge(src, tgt, weight=sim)
    return G

def reduce_to_2d(emb_norm, method="pca"):
    if method == "umap":
        try:
            import umap
        except Exception as e:
            raise RuntimeError("UMAP requested but umap-learn not installed. Install with `pip install umap-learn`.")
        reducer = umap.UMAP(n_components=2, n_neighbors=UMAP_N_NEIGHBORS, metric="cosine", random_state=42)
        coords = reducer.fit_transform(emb_norm)
    else:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(emb_norm)
    return coords

def plot_graph(G, ids, coords2d, docs, output_png, show_labels_threshold):
    pos = {ids[i]: tuple(coords2d[i]) for i in range(len(ids))}
    plt.figure(figsize=(12, 9))
    ax = plt.gca()
    # edges
    nx.draw_networkx_edges(G, pos=pos, alpha=0.25, width=0.7)
    # nodes (size by degree)
    degrees = np.array([G.degree(n) for n in ids])
    if degrees.max() == degrees.min():
        node_sizes = 80
    else:
        node_sizes = 50 + 200 * (degrees - degrees.min()) / (degrees.max() - degrees.min())
    nx.draw_networkx_nodes(G, pos=pos, node_size=node_sizes, node_color=node_sizes, cmap=plt.cm.viridis)
    # labels if small
    if len(ids) <= show_labels_threshold:
        labels = {ids[i]: (docs[i][:60] + "...") if docs[i] else ids[i] for i in range(len(ids))}
        nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=7)
    plt.title(f"k-NN graph (k={K}) from Chroma collection '{COLLECTION_NAME}' — 2D projection")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    print(f"Saved graph image to: {output_png}")

def save_node_table(G, ids, docs, metas, output_csv):
    rows = []
    for i, node_id in enumerate(ids):
        rows.append({
            "id": node_id,
            "degree": G.degree(node_id),
            "doc_sample": (docs[i][:240] if docs[i] else None),
            "metadata": metas[i]
        })
    df = pd.DataFrame(rows).sort_values("degree", ascending=False).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved node table to: {output_csv}")
    return df

def main():
    print("Opening collection...")
    col = load_collection(PERSIST_DIR, COLLECTION_NAME)
    print("Fetching embeddings and metadata (this may take a moment)...")
    embeddings, ids, docs, metas = fetch_embeddings_and_meta(col)
    print(f"Loaded {embeddings.shape[0]} vectors, dimension {embeddings.shape[1]}.")

    emb_norm = normalize_embeddings(embeddings)
    print("Computing k-NN graph...")
    G = build_knn_graph(emb_norm, ids, K)

    method = "umap" if USE_UMAP else "pca"
    print(f"Reducing to 2D using {method}...")
    coords2d = reduce_to_2d(emb_norm, method=method)

    print("Plotting graph...")
    plot_graph(G, ids, coords2d, docs, OUTPUT_PNG, SHOW_LABELS_THRESHOLD)
    df = save_node_table(G, ids, docs, metas, OUTPUT_CSV)

    # show top 10 high-degree nodes
    print("\nTop 10 nodes by degree:")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
