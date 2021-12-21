import os
import torch
from vilt.modules import ViLTransformerSS
from typing import List
import time

from transformers import BertTokenizer

DATASET_ROOT_PATH = os.getenv('DATASET_ROOT_PATH', '/hdd/master/tfm/flickr30k_images')
# The root path where the flickr30k entities per split is kept
DATASET_SPLIT_ROOT_PATH = os.getenv('DATASET_SPLIT_ROOT_PATH', '/hdd/master/tfm/flickr30k_images/flickr30k_entities')


class ViltModel(ViLTransformerSS):
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self._device = torch.device(dev)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.eval()
        self.to(self._device)

    def rank(self, query: str, images: List):
        rank_scores = []
        encoded_input = self.tokenizer(query, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        mask = encoded_input['attention_mask']
        batch = {"text_ids": input_ids.to(self.device), 'text_masks': mask.to(self._device), 'text_labels': None}
        # no masking
        for image in images:
            batch['image'] = [image.to(self._device).unsqueeze(0)]
            score = self.rank_output(self(batch)['cls_feats']).squeeze()
            rank_scores.append(score.detach().cpu().item())
        return rank_scores


if __name__ == '__main__':
    import copy
    from vilt import config
    from src.dataset.dataset import get_image_data_loader, get_captions_data_loader
    from src.evaluate import evaluate

    # Scared config is immutable object, so you need to deepcopy it.
    conf = copy.deepcopy(config.config())
    conf['load_path'] = 'vilt_irtr_f30k.ckpt'
    conf['test_only'] = True

    # You need to properly configure loss_names to initialize heads (0.5 means it initializes head, but ignores the
    # loss during training)
    loss_names = {
        'itm': 0.5,
        'mlm': 0,
        'mpp': 0,
        'vqa': 0,
        'imgcls': 0,
        'nlvr2': 0,
        'irtr': 1,
        'arc': 0,
    }
    conf['loss_names'] = loss_names

    vilt = ViltModel(conf)

    image_dataset = get_image_data_loader(root=DATASET_ROOT_PATH,
                                          split_root=DATASET_SPLIT_ROOT_PATH,
                                          split='test')

    text_dataset = get_captions_data_loader(root=DATASET_ROOT_PATH,
                                            split_root=DATASET_SPLIT_ROOT_PATH,
                                            split='test',
                                            batch_size=1)

    images = []
    filenames = []
    for filenames_batch, images_batch in image_dataset:
        filenames.extend(filenames_batch)
        images.extend(images_batch)

    retrieved_image_filenames = []
    groundtruth_expected_image_filenames = []
    print(f' number of queries {len(text_dataset)}, against {len(images)}')
    for matching_filename, query in text_dataset:
        filename = matching_filename[0]
        groundtruth_expected_image_filenames.append([filename])
        q = query[0]
        start = time.time()
        scores = vilt.rank(q, images)
        print(f' rank a query against all images in {time.time() - start}s ')
        retrieved_image_filenames.append([f for _, f in sorted(zip(scores, filenames))])

    evaluate(['recall', 'reciprocal_rank'], retrieved_image_filenames,
             groundtruth_expected_image_filenames,
             [1, 5, 10, 20, 100, 200, 500, None],
             {}, print_results=True)
