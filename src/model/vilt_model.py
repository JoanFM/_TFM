from typing import List, Dict

import torch
from transformers import BertTokenizer

from vilt.modules import ViLTransformerSS


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
            dev = "cuda"
        else:
            dev = "cpu"
        self._device = torch.device(dev)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.eval()

    @property
    def in_cuda(self):
        return next(self.parameters()).is_cuda

    def rank_query_vs_images(self, query: str, images: List):
        rank_scores = []
        encoded_input = self.tokenizer(query, return_tensors='pt')
        input_ids = encoded_input['input_ids'][:, :self._config['max_text_len']]
        mask = encoded_input['attention_mask'][:, :self._config['max_text_len']]
        in_cuda = self.in_cuda
        if in_cuda:
            input_ids = input_ids.to(self._device)
            mask = mask.to(self._device)
        batch = {'text_ids': input_ids, 'text_masks': mask, 'text_labels': None}
        # no masking
        for image in images:
            if in_cuda:
                image = image.to(self._device)
            batch['image'] = [image.unsqueeze(0)]
            score = self.rank_output(self.infer(batch)['cls_feats'])[:, 0]
            rank_scores.append(score.detach().cpu().item())
        return rank_scores

    def tokenize_texts(self, texts: List[str]):
        tokenized_texts = self.tokenizer(texts,
                                         return_tensors='pt',
                                         padding='max_length',
                                         max_length=self._config['max_text_len'],
                                         truncation=True)
        in_cuda = self.in_cuda
        if in_cuda:
            tokenized_texts['input_ids'] = tokenized_texts['input_ids'].to(self._device)
            tokenized_texts['attention_mask'] = tokenized_texts['attention_mask'].to(self._device)
        texts_dict = {
            'text_ids': tokenized_texts['input_ids'],
            'text_masks': tokenized_texts['attention_mask'],
            'text_labels': [None]
        }
        return texts_dict

    def score_image_vs_texts(self, image: torch.Tensor, texts: List[str]):
        texts_dict = self.tokenize_texts(texts)
        return self.score_image_vs_tokenized_texts(image, texts_dict)

    def score_image_vs_tokenized_texts(self, image: torch.Tensor, texts_dict: List[str]):
        fblen = len(texts_dict['text_ids'])
        (ie, im, _, _) = self.transformer.visual_embed(
            image.unsqueeze(0),
            max_image_len=-1,
            mask_it=False,
        )
        _, l, c = ie.shape
        ie = ie.expand(fblen, l, c)
        im = im.expand(fblen, l)

        scores = self.rank_output(
            self.infer(
                texts_dict,
                image_embeds=ie,
                image_masks=im,
            )['cls_feats']
        )[:, 0]

        return scores

    def score_query_vs_images(self, query: str, images: List):
        rank_scores = []
        encoded_input = self.tokenizer(query, return_tensors='pt')
        input_ids = encoded_input['input_ids'][:, :self._config['max_text_len']]
        mask = encoded_input['attention_mask'][:, :self._config['max_text_len']]
        in_cuda = self.in_cuda
        if in_cuda:
            input_ids = input_ids.to(self._device)
            mask = mask.to(self._device)
        batch = {'text_ids': input_ids, 'text_masks': mask, 'text_labels': None}
        # no masking
        for image in images:
            if in_cuda:
                image = image.to(self._device)
            batch['image'] = [image.unsqueeze(0)]
            score = self.rank_output(self.infer(batch)['cls_feats'])
            rank_scores.append(score[:, 0])
        return torch.stack(rank_scores, dim=1)[0]


def get_vilt_model(load_path=None):
    import copy
    from vilt import config

    conf = copy.deepcopy(config.config())
    conf['load_path'] = load_path or 'vilt_irtr_f30k.ckpt'
    conf['test_only'] = True
    conf['max_text_len'] = 40
    conf['max_text_len'] = 40
    conf['data_root'] = '/hdd/master/tfm/arrow'
    conf['datasets'] = ['f30k']
    conf['batch_size'] = 1
    conf['per_gpu_batchsize'] = 1
    conf['draw_false_image'] = 0
    conf['num_workers'] = 1

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

    return ViltModel(conf).train(False)
