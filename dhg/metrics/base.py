import abc
from functools import partial
from typing import Union, List, Dict
from collections import defaultdict

import numpy as np
import torch

from dhg._global import AUTHOR_EMAIL


def format_metric_configs(task: str, metric_configs: List[Union[str, Dict[str, dict]]]):
    r"""Format metric_configs.
    
    Args:
        ``task`` (``str``): The type of the task. The supported types include: ``classification``, ``retrieval`` and ``recommender``.
        ``metric_configs`` (``Dict[str, Dict[str, Union[str, int]]]``): The metric configs.
    """
    task = task.lower()
    if task == "classification":
        import dhg.metrics.classification as module

        available_metrics = module.available_classification_metrics()
    elif task == "retrieval":
        import dhg.metrics.retrieval as module

        available_metrics = module.available_retrieval_metrics()
    elif task == "recommender":
        import dhg.metrics.recommender as module

        available_metrics = module.available_recommender_metrics()
    else:
        raise ValueError(f"Task {task} is not supported yet. Please email '{AUTHOR_EMAIL}' to add it.")
    metric_list = []
    for metric in metric_configs:
        if isinstance(metric, str):
            marker, func_name = metric, metric
            assert (
                func_name in available_metrics
            ), f"{func_name} is not supported yet. Please email '{AUTHOR_EMAIL}' to add it."
            func = getattr(module, func_name)
        elif isinstance(metric, dict):
            assert len(metric) == 1
            func_name = list(metric.keys())[0]
            assert (
                func_name in available_metrics
            ), f"{func_name} is not supported yet. Please email '{AUTHOR_EMAIL}' to add it."
            params = metric[func_name]
            func = getattr(module, func_name)
            func = partial(func, **params)
            markder_list = []
            for k, v in params.items():
                _m = f"{k}@"
                if isinstance(v, str):
                    _m += v
                elif isinstance(v, int):
                    _m += str(v)
                elif isinstance(v, float):
                    _m += f"{v:.4f}"
                elif isinstance(v, list) or isinstance(v, tuple) or isinstance(v, set):
                    _m += "_".join([str(_v) for _v in v])
                else:
                    _m += str(v)
                markder_list.append(_m)
            marker = f"{func_name} -> {' | '.join(markder_list)}"
        else:
            raise ValueError
        metric_list.append({"marker": marker, "func": func, "func_name": func_name})
    return metric_list


class BaseEvaluator:
    r"""The base class for task-specified metric evaluators.
    
    Args:
        ``task`` (``str``): The type of the task. The supported types include: ``classification``, ``retrieval`` and ``recommender``.
        ``metric_configs`` (``List[Union[str, Dict[str, dict]]]``): The metric configurations. The key is the metric name and the value is the metric parameters.
        ``validate_index`` (``int``): The specified metric index used for validation. Defaults to ``0``.
    """

    def __init__(
        self, task: str, metric_configs: List[Union[str, Dict[str, dict]]], validate_index: int = 0,
    ):
        self.validate_index = validate_index
        metric_configs = format_metric_configs(task, metric_configs)
        assert validate_index >= 0 and validate_index < len(
            metric_configs
        ), "The specified validate metric index is out of range."
        self.marker_list, self.func_list = [], []
        for metric in metric_configs:
            self.marker_list.append(metric["marker"])
            self.func_list.append(metric["func"])
        # init batch data containers
        self.validate_res = []
        self.test_res_dict = defaultdict(list)
        self.last_validate_res, self.last_test_res = None, {}

    @abc.abstractmethod
    def __repr__(self) -> str:
        r"""Print the Evaluator information.
        """

    def validate_add_batch(self, batch_y_true: torch.Tensor, batch_y_pred: torch.Tensor):
        r"""Add batch data for validation.

        Args:
            ``batch_y_true`` (``torch.Tensor``): The ground truth data. Size :math:`(N_{batch}, -)`.
            ``batch_y_pred`` (``torch.Tensor``): The predicted data. Size :math:`(N_{batch}, -)`.
        """
        batch_res = self.func_list[self.validate_index](batch_y_true, batch_y_pred, ret_batch=True)
        batch_res = np.array(batch_res)
        if len(batch_res.shape) == 1:
            batch_res = batch_res[:, np.newaxis]
        self.validate_res.append(batch_res)

    def validate_epoch_res(self):
        r"""For all added batch data, return the result of the evaluation on the specified ``validate_index``-th metric.
        """
        if self.validate_res == [] and self.last_validate_res is not None:
            return self.last_validate_res
        assert self.validate_res != [], "No batch data added for validation."
        self.last_validate_res = np.vstack(self.validate_res).mean(0).item()
        # clear batch cache
        self.validate_res = []
        return self.last_validate_res

    def test_add_batch(self, batch_y_true: torch.Tensor, batch_y_pred: torch.Tensor):
        r"""Add batch data for testing.

        Args:
            ``batch_y_true`` (``torch.Tensor``): The ground truth data. Size :math:`(N_{batch}, -)`.
            ``batch_y_pred`` (``torch.Tensor``): The predicted data. Size :math:`(N_{batch}, -)`.
        """
        for name, func in zip(self.marker_list, self.func_list):
            batch_res = func(batch_y_true, batch_y_pred, ret_batch=True)
            if not isinstance(batch_res, tuple):
                batch_res = np.array(batch_res)
                if len(batch_res.shape) == 1:
                    batch_res = batch_res[:, np.newaxis]
                self.test_res_dict[name].append(batch_res)
            else:
                if self.test_res_dict[name] == []:
                    self.test_res_dict[name] = [list() for _ in range(len(batch_res))]
                for idx, batch_sub_res in enumerate(batch_res):
                    batch_sub_res = np.array(batch_sub_res)
                    if len(batch_sub_res.shape) == 1:
                        batch_sub_res = batch_sub_res[:, np.newaxis]
                    self.test_res_dict[name][idx].append(batch_sub_res)

    def test_epoch_res(self):
        r"""For all added batch data, return results of the evaluation on all the metrics in ``metric_configs``.
        """
        if self.test_res_dict == {} and self.last_test_res is not None:
            return self.last_test_res
        assert self.test_res_dict != {}, "No batch data added for testing."
        for name, res_list in self.test_res_dict.items():
            if not isinstance(res_list[0], list):
                self.last_test_res[name] = np.vstack(res_list).mean(0).squeeze().tolist()
            else:
                self.last_test_res[name] = [
                    np.vstack(sub_res_list).mean(0).squeeze().tolist() for sub_res_list in res_list
                ]
        # clear batch cache
        self.test_res_dict = defaultdict(list)
        return self.last_test_res

    def validate(self, y_true: torch.LongTensor, y_pred: torch.Tensor):
        r"""Return the result of the evaluation on the specified ``validate_index``-th metric.

        Args:
            ``y_true`` (``torch.LongTensor``): The ground truth labels. Size :math:`(N_{samples}, -)`.
            ``y_pred`` (``torch.Tensor``): The predicted labels. Size :math:`(N_{samples}, -)`.
        """
        return self.func_list[self.validate_index](y_true, y_pred)

    def test(self, y_true: torch.LongTensor, y_pred: torch.Tensor):
        r"""Return results of the evaluation on all the metrics in ``metric_configs``.

        Args:
            ``y_true`` (``torch.LongTensor``): The ground truth labels. Size :math:`(N_{samples}, -)`.
            ``y_pred`` (``torch.Tensor``): The predicted labels. Size :math:`(N_{samples}, -)`.
        """
        return {name: func(y_true, y_pred) for name, func in zip(self.marker_list, self.func_list)}

