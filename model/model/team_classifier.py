from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from random import shuffle

import umap
from sklearn.cluster import KMeans

from core import settings
from model.embedding import Embedding


class TeamClassifier(Embedding):
    def __init__(self, model_name: str = settings.TEAM_CLASSIFIER_MODEL_NAME, batch_size: int = settings.BATCH_SIZE):
        super().__init__(model_name=model_name, batch_size=batch_size)
        self.__umap = umap.UMAP(n_components=3)
        self.__kmeans = KMeans(n_clusters=2)

    @staticmethod
    def build_dataset(data_path: str, stride: int = 1):
        from model.inference import Inference

        paths = list(Path(__file__).parent.parent.glob(f"{data_path}/*.mp4"))
        inference = Inference("detection.pt")

        with ThreadPoolExecutor() as executor:
            crops = list(executor.map(lambda f: inference.crop(f, stride), paths))

        dataset = [item for sublist in crops for item in sublist]
        shuffle(dataset)

        return dataset

    def fit(self, dataset: list):
        shuffle(dataset)
        train_embeddings = self.get_embeddings(dataset)
        projections = self.__umap.fit_transform(train_embeddings)
        self.__kmeans.fit(projections)

    def predict(self, dataset: list):
        embeddings = self.get_embeddings(dataset)
        projections = self.__umap.transform(embeddings)
        return self.__kmeans.predict(projections)


if __name__ == "__main__":
    classifier = TeamClassifier()
    data = classifier.build_dataset("tests/", 120)
    classifier.fit(data)
    predictions = classifier.predict(data)

    for i, pred in enumerate(predictions):
        print(f"Video {i}: {'Team A' if pred == 0 else 'Team B'}")
