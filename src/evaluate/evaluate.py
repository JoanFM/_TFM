from typing import Sequence, Any, Optional, List, Dict
from collections import defaultdict


def recall(
        actual: Sequence[Any], desired: Sequence[Any], eval_at: Optional[int], *args, **kwargs
) -> float:
    """
    Compute recall, understanding recall as 1 if expected is found in the first {eval_at} list
    """
    if eval_at == 0:
        return 0.0
    actual_at_k = actual[: eval_at] if eval_at else actual
    return 1.0 if desired[0] in actual_at_k else 0.0


def reciprocal_rank(
        actual: Sequence[Any], desired: Sequence[Any], *args, **kwargs
) -> float:
    """
    Evaluate score as per reciprocal rank metric.
    :param actual: Sequence of sorted document IDs.
    :param desired: Sequence of sorted relevant document IDs
        (the first is the most relevant) and the one to be considered.
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    :return: Reciprocal rank score
    """
    if len(actual) == 0 or len(desired) == 0:
        return 0.0
    try:
        return 1.0 / (actual.index(desired[0]) + 1)
    except:
        return 0.0


def evaluate(metrics: List[str],
             retrieved_image_filenames: List[List[str]],
             groundtruth_expected_image_filenames: List[List[str]],
             top_ks: List[Optional[int]],
             accum_evaluation_results: Optional[Dict] = None,
             print_results: bool = True):
    ret = {}
    for metric in metrics:
        if metric == 'num_candidates':
            mean_num_candidates = 0
            for actual in retrieved_image_filenames:
                mean_num_candidates += len(actual)
                if accum_evaluation_results is not None:
                    if 'num_candidates' not in accum_evaluation_results:
                        accum_evaluation_results['num_candidates'] = defaultdict(float)
                    accum_evaluation_results['num_candidates']['sum'] += len(actual)
                    accum_evaluation_results['num_candidates']['num'] += 1
            ret[f'num_candidates'] = mean_num_candidates/len(retrieved_image_filenames)
            if print_results:
                print(f' Mean num_candidates {mean_num_candidates/len(retrieved_image_filenames)}')
        if metric == 'recall':
            for eval_at in top_ks:
                mean_recall = 0
                for actual, desired in zip(retrieved_image_filenames, groundtruth_expected_image_filenames):
                    recall_value = recall(actual, desired, eval_at)
                    mean_recall += recall_value
                    if accum_evaluation_results is not None:
                        if f'recall@{eval_at}' not in accum_evaluation_results:
                            accum_evaluation_results[f'recall@{eval_at}'] = defaultdict(float)
                        accum_evaluation_results[f'recall@{eval_at}']['sum'] += recall_value
                        accum_evaluation_results[f'recall@{eval_at}']['num'] += 1
                ret[f'recall@{eval_at}'] = mean_recall/len(groundtruth_expected_image_filenames)
                if print_results:
                    print(f' Mean Recall@{eval_at} {mean_recall/len(groundtruth_expected_image_filenames)}')
        if metric == 'reciprocal_rank':
            mean_reciprocal_rank = 0
            for actual, desired in zip(retrieved_image_filenames, groundtruth_expected_image_filenames):
                reciprocal_rank_value = reciprocal_rank(actual, desired)
                mean_reciprocal_rank += reciprocal_rank_value
                if accum_evaluation_results is not None:
                    if f'reciprocal_rank' not in accum_evaluation_results:
                        accum_evaluation_results[f'reciprocal_rank'] = defaultdict(float)
                    accum_evaluation_results[f'reciprocal_rank']['sum'] += reciprocal_rank_value
                    accum_evaluation_results[f'reciprocal_rank']['num'] += 1
            if print_results:
                print(f' Mean Reciprocal rank {mean_reciprocal_rank/len(groundtruth_expected_image_filenames)}')
            ret['reciprocal_rank'] = mean_reciprocal_rank/len(groundtruth_expected_image_filenames)
    return ret
