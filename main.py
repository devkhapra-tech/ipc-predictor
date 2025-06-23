import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
# Load IPC info
ipc_df = pd.read_csv("ipc_sections.csv")  # Use the IPC Kaggle dataset

# Clean
ipc_df = ipc_df.dropna(subset=["Section", "Description"])
ipc_df["Section_Text"] = "Section " + ipc_df["Section"].astype(str) + ": " + ipc_df["Description"]

# Load BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
ipc_embeddings = model.encode(ipc_df["Section_Text"].tolist(), convert_to_tensor=True)

def predict(crime_text,top_k):
    user_embedding = model.encode([crime_text], convert_to_tensor=True)
    similar = cosine_similarity(user_embedding.cpu().numpy(), ipc_embeddings.cpu().numpy())[0]

    top_indices=torch.topk(torch.tensor(similar),k=top_k,dim=0).indices
    results = []
    for idx in top_indices:
        idx=int(idx)
        results.append({
            "section": ipc_df.iloc[idx]["Section_Text"],
            "score": float(similar[idx])
        })
    return results

