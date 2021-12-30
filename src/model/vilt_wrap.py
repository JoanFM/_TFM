import os
import time
from src.model.vilt_model import ViltModel

DATASET_ROOT_PATH = os.getenv('DATASET_ROOT_PATH', '/hdd/master/tfm/flickr30k_images')
# The root path where the flickr30k entities per split is kept
DATASET_SPLIT_ROOT_PATH = os.getenv('DATASET_SPLIT_ROOT_PATH', '/hdd/master/tfm/flickr30k_images/flickr30k_entities')


# if __name__ == '__main__':
#     import copy
#     from vilt import config
#     from vilt.transforms.pixelbert import pixelbert_transform
#     from src.dataset.dataset import get_image_data_loader, get_captions_data_loader
#     from src.evaluate import evaluate
#
#     # Scared config is immutable object, so you need to deepcopy it.
#     conf = copy.deepcopy(config.config())
#     print(f' conf {conf}')
#     conf['load_path'] = 'vilt_irtr_f30k.ckpt'
#     conf['test_only'] = True
#     conf['max_text_len'] = 40
#     conf['max_text_len'] = 40
#     conf['data_root'] = '/hdd/master/tfm/arrow'
#     conf['datasets'] = ['f30k']
#     conf['batch_size'] = 1
#     conf['per_gpu_batchsize'] = 1
#     conf['draw_false_image'] = 0
#     conf['num_workers'] = 1
#
#     # You need to properly configure loss_names to initialize heads (0.5 means it initializes head, but ignores the
#     # loss during training)
#     loss_names = {
#         'itm': 0.5,
#         'mlm': 0,
#         'mpp': 0,
#         'vqa': 0,
#         'imgcls': 0,
#         'nlvr2': 0,
#         'irtr': 1,
#         'arc': 0,
#     }
#     conf['loss_names'] = loss_names
#
#     vilt_model = ViltModel(conf)
#
#     image_dataset = get_image_data_loader(root=DATASET_ROOT_PATH,
#                                           split_root=DATASET_SPLIT_ROOT_PATH,
#                                           split='test',
#                                           transform=pixelbert_transform(384),
#                                           batch_size=1)
#
#     text_dataset = get_captions_data_loader(root=DATASET_ROOT_PATH,
#                                             split_root=DATASET_SPLIT_ROOT_PATH,
#                                             split='test',
#                                             batch_size=1)
#
#     images = []
#     filenames = []
#     for filenames_batch, images_batch in image_dataset:
#         filenames.extend(filenames_batch)
#         images.extend(images_batch)
#
#     retrieved_image_filenames = []
#     groundtruth_expected_image_filenames = []
#     print(f' number of queries {len(text_dataset)}, against {len(images)}')
#     for matching_filename, query in text_dataset:
#         filename = matching_filename[0]
#         groundtruth_expected_image_filenames.append([filename])
#         q = query[0]
#         start = time.time()
#         scores = vilt_model.rank(q, images)
#         retrieved_image_filenames.append([f for _, f in sorted(zip(scores, filenames), reverse=True)])
#         print(
#             f' matching_filename {matching_filename[0]} in {retrieved_image_filenames[-1].index(matching_filename[0])}')
#
#     evaluate(['recall', 'reciprocal_rank'], retrieved_image_filenames,
#              groundtruth_expected_image_filenames,
#              [1, 5, 10, 20, 100, 200, 500, None],
#              {}, print_results=True)
def compute_irtr_recall(pl_module):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import os
    import glob
    import json
    import tqdm
    import functools
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    print(f' text_dset {type(text_dset)}')
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=8,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    # dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        # sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        print(f'shape {_b["text_ids"].shape}')
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )
    print(f' length text_preload {len(text_preload)}')
    print(f' type text_ids {type(text_preload[0]["text_ids"])}')
    print(f' shape text_ids {text_preload[0]["text_ids"].shape}')

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)
    print(f' tiids shape {tiids.shape}')

    image_preload = list()
    print(f' max_image_len {pl_module.hparams.config["max_image_len"]}')
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        print(f' _b["image"][0] shape {_b["image"][0].shape}')

        (ie, im, _, _) = pl_module.transformer.visual_embed(
            _b["image"][0].to(pl_module.device),
            max_image_len=pl_module.hparams.config["max_image_len"],
            mask_it=False,
        )
        print(f' ie shape {ie.shape}')
        image_preload.append((ie, im, _b["img_index"][0]))

    print(f' length image_preload {len(image_preload)}')
    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _ie, _im, _iid = img_batch
        _, l, c = _ie.shape
        print(f' img_batch {type(img_batch)}')
        print(f' _ie.shape {_ie.shape}')

        img_batch_score = list()
        for txt_batch in text_preload:
            print(f' txt_batch {txt_batch["text_ids"].shape}')
            fblen = len(txt_batch["text_ids"])
            print(f' fblen {fblen}')
            ie = _ie.expand(fblen, l, c)
            im = _im.expand(fblen, l)

            with torch.cuda.amp.autocast():
                input_batch = {
                    "text_ids": txt_batch["text_ids"],
                    "text_masks": txt_batch["text_masks"],
                    "text_labels": txt_batch["text_labels"],
                }
                print(f' input_batch {input_batch["text_ids"].shape}')
                print(f' image_embs shape {ie.shape}')
                score = pl_module.rank_output(
                    pl_module.infer(
                        input_batch,
                        image_embeds=ie,
                        image_masks=im,
                    )["cls_feats"]
                )[:, 0]
                print(f' score {score}')

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        print(f' img_batch_score {img_batch_score.shape}')
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10


if __name__ == '__main__':
    import copy
    from vilt import config
    from vilt.transforms.pixelbert import pixelbert_transform
    from src.dataset.dataset import get_image_data_loader, get_captions_data_loader
    from src.evaluate import evaluate
    from vilt.datamodules.multitask_datamodule import MTDataModule

    # Scared config is immutable object, so you need to deepcopy it.
    conf = copy.deepcopy(config.config())
    print(f' conf {conf}')
    conf['load_path'] = 'vilt_irtr_f30k.ckpt'
    conf['test_only'] = True
    conf['max_text_len'] = 40
    conf['data_root'] = '/hdd/master/tfm/arrow'
    conf['datasets'] = ['f30k']
    conf['batch_size'] = 1
    conf['per_gpu_batchsize'] = 1
    conf['draw_false_image'] = 0
    conf['num_workers'] = 1

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

    dm = MTDataModule(conf, dist=False)
    dm.setup(stage='test')


    class A:
        datamodule = dm


    vilt_model = ViltModel(conf)
    vilt_model.trainer = A()

    compute_irtr_recall(vilt_model)

    test_dataloader = dm.test_dataloader()

    for batch in test_dataloader:
        print(f' keys {batch.keys()}')
        for image in batch["image"]:
            print(f' image {image.shape}')
        break
