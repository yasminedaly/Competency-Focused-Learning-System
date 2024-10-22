from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SimilarityCalculator:
    def __init__(self, model='sentence-transformers/all-MiniLM-L6-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model).to(self.device)

    def calculate_similarity(self, df1, df2):
        similarity_scores = []
        for activity in df1.processed_activities:
            activity = str(activity)  # Convert activity to string
            activity_embeddings = self.model.encode(activity, convert_to_tensor=True)
            activity_similarity_scores = []
            for col in df2.columns:
                col_embeddings = self.model.encode(str(df2[col]), convert_to_tensor=True)
                similarity = cosine_similarity(activity_embeddings.reshape(1, -1), col_embeddings.reshape(1, -1))
                activity_similarity_scores.append(similarity.item())
            similarity_scores.append(activity_similarity_scores)
        return similarity_scores


class ActivityClustering:
    def __init__(self, model='sentence-transformers/all-MiniLM-L6-v2'):
        self.similarity_calculator = SimilarityCalculator(model)

    def cluster_activities(self, df, specialties_df):
        similarity_scores = self.similarity_calculator.calculate_similarity(df, specialties_df)

        max_sim_col = []
        for sim_scores in similarity_scores:
            max_sim_col.append(specialties_df.columns[np.argmax(sim_scores)])

        df['controlled_dic'] = max_sim_col

        df = df.sort_values('controlled_dic')
        df.to_csv("finalcluster.csv", index=False)
        return df


# Example usage:
# activity_clustering = ActivityClustering()

# df = pd.read_csv("df.csv")
# specialties_df = pd.read_csv("specialties.csv")

# clustered_df = activity_clustering.cluster_activities(df, specialties_df)
