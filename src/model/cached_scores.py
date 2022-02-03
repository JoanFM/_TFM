import torch


class CachedScores:

    def __init__(self, base_path, number_partitions, number_captions):
        self.base_path = base_path
        self.number_partitions = number_partitions
        self.number_captions = number_captions
        self._loaded_partition = None
        self._loaded_cache = None

    @property
    def loaded_cache_scores(self):
        return self._loaded_cache

    @loaded_cache_scores.setter
    def loaded_cache_scores(self, partition):
        path = self.base_path if self.number_partitions == 1 else f'{self.base_path.split("th")}-{partition}.th'
        self._loaded_cache = torch.load(path)

    def get_scores(self, caption_id, images_ids):
        extract_partiton, offset_in_partition = divmod(caption_id, (self.number_captions // self.number_partitions))
        if self._loaded_partition is not None and self._loaded_partition != extract_partiton:
            if self._loaded_cache is not None:
                del self._loaded_cache
            self.loaded_cache_scores = extract_partiton
        self._loaded_partition = extract_partiton
        return self.loaded_cache_scores[offset_in_partition][images_ids]
