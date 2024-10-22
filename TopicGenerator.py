import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import glob


class TopicGenerator:
    def __init__(self):
        self.topics = {}

    def generate_topics_csv(self, data):
        # Create a bag of words matrix using CountVectorizer
        vectorizer = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1, 3), analyzer='word')
        top_words_dict = {}
        for title, activities in data.items():
            bow_matrix = vectorizer.fit_transform(activities)

            # Train an LDA model on the bag of words matrix
            lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
            lda_model.fit(bow_matrix)

            # Extract the top words associated with each topic
            feature_names = vectorizer.get_feature_names_out()
            top_words_per_topic = []
            for topic_idx, topic in enumerate(lda_model.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-5 - 1:-1]]
                top_words_per_topic.append(top_words)

            # Store the top words for the current title
            top_words_dict[title] = top_words_per_topic

        return top_words_dict

# Example usage:
# topic_generator = TopicGenerator()
# grouped_data = data.groupby('controlled_dic')
# topic_generator.process_grouped_data(grouped_data)
