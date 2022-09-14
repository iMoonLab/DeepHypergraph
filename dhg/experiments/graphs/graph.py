from pathlib import Path
from typing import Callable, Optional, Union
from ..vertex_classification import VertexClassificationTask

import torch
import torch.nn as nn
import optuna

from dhg.metrics import BaseEvaluator


class GraphVertexClassificationTask(VertexClassificationTask):
    r"""The auto-experiment class for the vertex classification task on graph.

    Args:
        ``work_root`` (``Optional[Union[str, Path]]``): User's work root to store all studies.
        ``data`` (``dict``): The dictionary to store input data that used in the experiment.
        ``model_builder`` (``Callable``): The function to build a model with a fixed parameter ``trial``.
        ``train_builder`` (``Callable``): The function to build a training configuration with two fixed parameters ``trial`` and ``model``.
        ``evaluator`` (``dhg.metrics.BaseEvaluator``): The DHG evaluator object to evaluate performance of the model in the experiment.
        ``device`` (``torch.device``): The target device to run the experiment.
        ``structure_builder`` (``Optional[Callable]``): The function to build a structure with a fixed parameter ``trial``. The structure should be ``dhg.Graph``.
        ``study_name`` (``Optional[str]``): The name of this study. If set to ``None``, the study name will be generated automatically according to current time. Defaults to ``None``.
        ``overwrite`` (``bool``): The flag that whether to overwrite the existing study. Different studies are identified by the ``study_name``. Defaults to ``True``.
    """
    def __init__(
        self,
        work_root: Optional[Union[str, Path]],
        data: dict,
        model_builder: Callable,
        train_builder: Callable,
        evaluator: BaseEvaluator,
        device: torch.device,
        study_name: Optional[str] = None,
        overwrite: bool = True,
    ):
        super().__init__(
            work_root,
            data,
            model_builder,
            train_builder,
            evaluator,
            device,
            study_name=study_name,
            overwrite=overwrite,
        )

    def to(self, device: torch.device):
        r"""Move the input data to the target device.

        Args:
            ``device`` (``torch.device``): The specified target device to store the input data.
        """
        return super().to(device)

    @property
    def vars_for_DL(self):
        r"""Return a name list for available variables for deep learning in the vertex classification on graph. The name list includes ``features``, ``structure``, ``labels``, ``train_mask``, ``val_mask``, and ``test_mask``.
        """
        return super().vars_for_DL

    def experiment(self, trial: optuna.Trial):
        r"""Run the experiment for a given trial.

        Args:
            ``trial`` (``optuna.Trial``): The ``optuna.Trial`` object.
        """
        return super().experiment(trial)

    def run(self, max_epoch: int, num_trials: int = 1, direction: str = "maximize"):
        r"""Run experiments with automatically hyper-parameter tuning.

        Args:
            ``max_epoch`` (``int``): The maximum number of epochs to train for each experiment.
            ``num_trials`` (``int``): The number of trials to run. Defaults to ``1``.
            ``direction`` (``str``): The direction to optimize. Defaults to ``"maximize"``.
        """
        return super().run(max_epoch, num_trials, direction)

    def train(
        self,
        data: dict,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ):
        r"""Train model for one epoch.

        Args:
            ``data`` (``dict``): The input data.
            ``model`` (``nn.Module``): The model.
            ``optimizer`` (``torch.optim.Optimizer``): The model optimizer.
            ``criterion`` (``nn.Module``): The loss function.
        """
        return super().train(data, model, optimizer, criterion)

    @torch.no_grad()
    def validate(self, data: dict, model: nn.Module):
        r"""Validate the model.

        Args:
            ``data`` (``dict``): The input data.
            ``model`` (``nn.Module``): The model.
        """
        return super().validate(data, model)

    @torch.no_grad()
    def test(self, data: Optional[dict] = None, model: Optional[nn.Module] = None):
        r"""Test the model.

        Args:
            ``data`` (``dict``, optional): The input data if set to ``None``, the specified ``data`` in the intialization of the experiments will be used. Defaults to ``None``.
            ``model`` (``nn.Module``, optional): The model if set to ``None``, the trained best model will be used. Defaults to ``None``.
        """
        return super().test(data, model)
