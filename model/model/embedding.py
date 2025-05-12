import numpy as np
import torch
from more_itertools import chunked
from transformers import AutoProcessor, SiglipVisionModel

from core import settings


class Embedding:
    def __init__(self, model_name: str = settings.TEAM_CLASSIFIER_MODEL_NAME, batch_size: int = settings.BATCH_SIZE):
        self.__processor = AutoProcessor.from_pretrained(model_name)
        self.__siglip_model = SiglipVisionModel.from_pretrained(model_name).to(settings.DEVICE)
        self.__batch_size = batch_size

    def get_embeddings(self, images: list):
        data = []
        with torch.no_grad():
            for batch in chunked(images, self.__batch_size):
                inputs = self.__processor(images=batch, return_tensors="pt").to(settings.DEVICE)
                inputs = {k: v.to(settings.DEVICE) for k, v in inputs.items()}
                outputs = self.__siglip_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)
        return np.concatenate(data, axis=0)
