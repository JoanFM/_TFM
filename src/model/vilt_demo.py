from vilt import config;
from src.dataset import get_image_data_loader;
from vilt.modules import ViLTransformerSS;
import os

# DATASET_SPLIT_ROOT_PATH = os.getenv('DATASET_SPLIT_ROOT_PATH', '/hdd/master/tfm/flickr30k_images/flickr30k_entities');
# DATASET_ROOT_PATH = os.getenv('DATASET_ROOT_PATH', '/hdd/master/tfm/flickr30k_images');
#
# split = 'test';
# root = DATASET_ROOT_PATH;
# split_root = DATASET_SPLIT_ROOT_PATH
# batch_size = 16;
# image_data_loader = get_image_data_loader(root=root, split_root=split_root, split=split, batch_size=batch_size);
# images = list(image_data_loader)[0]

import torch
import copy
from vilt import config
from vilt.modules import ViLTransformerSS

# Scared config is immutable object, so you need to deepcopy it.
conf = copy.deepcopy(config.config())
# conf["load_path"] = 'vilt_irtr_f30k.ckpt'
# conf["test_only"] = True
# conf['data_root'] =
#
# # You need to properly configure loss_names to initialize heads (0.5 means it initializes head, but ignores the loss during training)
# loss_names = {
#     "itm": 0.5,
#     "mlm": 0,
#     "mpp": 0,
#     "vqa": 0,
#     "imgcls": 0,
#     "nlvr2": 0,
#     "irtr": 1,
#     "arc": 0,
# }
# conf["loss_names"] = loss_names
#
# # two different random images
# image_vilt = torch.ones(1, 3, 224, 224)
# batch = {}
# batch["image"] = [image_vilt]
# # repeated random sentence tokens
# batch["text_ids"] = torch.IntTensor([[1, 0, 16, 32, 55]])
# # no masking
# batch["text_masks"] = torch.IntTensor([[1, 1, 1, 1, 1]])
# batch["text_labels"] = None
#
# with torch.no_grad():
#     vilt = ViLTransformerSS(conf)
#     vilt.train(mode=False)
#     out = vilt(batch)
#     #
#     # itm_logit = vilt.itm_score(out["cls_feats"]).squeeze()
#     # print(
#     #     f"itm logit, two logit (logit_neg, logit_pos) for a image-text pair.\n{itm_logit}"
#     # )
#     # itm_score = itm_logit.softmax(dim=-1)[:, 1]
#     # print(f"itm score, one score for a image-text pair.\n{itm_score}")
#     # print(f"a.\n{itm_logit.softmax(dim=-1)}")
#
#     # You should see "rank_output" head if loss_names["irtr"] > 0
#     score = vilt.rank_output(out["cls_feats"]).squeeze()
#     print(f"unnormalized irtr score, one score for a image-text pair.\n{score}")
#
#     normalized_score = score.softmax(dim=0)
#     print(
#         f"normalized (relative) irtr score, one score for a image-text pair.\n{score}"
#     )

from vilt.datamodules.f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataset

conf['data_root'] = '/hdd/master/tfm/arrow'
conf['data_dir'] = '/hdd/master/tfm/arrow'
conf['transform_keys'] = ['pixelbert']
conf['image_size'] = 384
#conf['names'] = 'f30k_caption_karpathy_test'
print(f' conf {conf}')
dataset = F30KCaptionKarpathyDataset(split='test', **conf)
print(f' dataset_ 0 {dataset[0]}')
