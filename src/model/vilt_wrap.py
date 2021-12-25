import os
import torch
from vilt.modules import ViLTransformerSS
from typing import List
import time

from transformers import BertTokenizer

DATASET_ROOT_PATH = os.getenv('DATASET_ROOT_PATH', '/hdd/master/tfm/flickr30k_images')
# The root path where the flickr30k entities per split is kept
DATASET_SPLIT_ROOT_PATH = os.getenv('DATASET_SPLIT_ROOT_PATH', '/hdd/master/tfm/flickr30k_images/flickr30k_entities')


def collate(batch, mlm_collator):
    batch_size = len(batch)
    keys = set([key for b in batch for key in b.keys()])
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
    img_sizes = list()

    for img_key in img_keys:
        img = dict_batch[img_key]
        img_sizes += [ii.shape for i in img if i is not None for ii in i]

    for size in img_sizes:
        assert (
                len(size) == 3
        ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

    if len(img_keys) != 0:
        max_height = max([i[1] for i in img_sizes])
        max_width = max([i[2] for i in img_sizes])

    for img_key in img_keys:
        img = dict_batch[img_key]
        view_size = len(img[0])

        new_images = [
            torch.zeros(batch_size, 3, max_height, max_width)
            for _ in range(view_size)
        ]

        for bi in range(batch_size):
            orig_batch = img[bi]
            for vi in range(view_size):
                if orig_batch is None:
                    new_images[vi][bi] = None
                else:
                    orig = img[bi][vi]
                    new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

        dict_batch[img_key] = new_images

    txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

    if len(txt_keys) != 0:
        texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        draw_text_len = len(encodings)
        flatten_encodings = [e for encoding in encodings for e in encoding]
        flatten_mlms = mlm_collator(flatten_encodings)

        for i, txt_key in enumerate(txt_keys):
            texts, encodings = (
                [d[0] for d in dict_batch[txt_key]],
                [d[1] for d in dict_batch[txt_key]],
            )

            mlm_ids, mlm_labels = (
                flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
            )

            input_ids = torch.zeros_like(mlm_ids)
            attention_mask = torch.zeros_like(mlm_ids)
            for _i, encoding in enumerate(encodings):
                _input_ids, _attention_mask = (
                    torch.tensor(encoding["input_ids"]),
                    torch.tensor(encoding["attention_mask"]),
                )
                input_ids[_i, : len(_input_ids)] = _input_ids
                attention_mask[_i, : len(_attention_mask)] = _attention_mask

            dict_batch[txt_key] = texts
            dict_batch[f"{txt_key}_ids"] = input_ids
            dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
            dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
            dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
            dict_batch[f"{txt_key}_masks"] = attention_mask

    return dict_batch


class ViltModel(ViLTransformerSS):
    def __init__(
            self,
            config,
            *args,
            **kwargs,
    ):
        super().__init__(config)
        self._config = config
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
        input_ids = encoded_input['input_ids'][:, :self._config['max_text_len']]
        mask = encoded_input['attention_mask'][:, :self._config['max_text_len']]
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
    from vilt.transforms.pixelbert import pixelbert_transform
    from src.dataset.dataset import get_image_data_loader, get_captions_data_loader
    from src.evaluate import evaluate

    # Scared config is immutable object, so you need to deepcopy it.
    conf = copy.deepcopy(config.config())
    print(f' conf {conf}')
    conf['load_path'] = 'vilt_irtr_f30k.ckpt'
    conf['test_only'] = True
    conf['max_text_len'] = 40

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

    vilt_model = ViltModel(conf)

    image_dataset = get_image_data_loader(root=DATASET_ROOT_PATH,
                                          split_root=DATASET_SPLIT_ROOT_PATH,
                                          split='test',
                                          transform=pixelbert_transform(384),
                                          batch_size=1)

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
        scores = vilt_model.rank(q, images)
        print(f' rank a query against all images in {time.time() - start}s ')
        retrieved_image_filenames.append([f for _, f in sorted(zip(scores, filenames))])

    evaluate(['recall', 'reciprocal_rank'], retrieved_image_filenames,
             groundtruth_expected_image_filenames,
             [1, 5, 10, 20, 100, 200, 500, None],
             {}, print_results=True)
