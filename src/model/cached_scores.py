import torch


class CachedScores:

    def __init__(self, load_path):
        self.load_path = load_path
        self._scores = torch.load(load_path)

    def get_scores(self, caption_id, images_ids):
        return self._scores[caption_id][images_ids]
