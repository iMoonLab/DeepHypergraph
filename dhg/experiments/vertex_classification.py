from typing import Callable, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
import optuna

from dhg.metrics import BaseEvaluator
from .base import BaseTask


class VertexClassificationTask(BaseTask):
    r"""The auto-experiment class for the vertex classification task.

    Args:
        ``work_root`` (``Optional[Union[str, Path]]``): User's work root to store all studies.
        ``data`` (``dict``): The dictionary to store input data that used in the experiment.
        ``model_builder`` (``Callable``): The function to build a model with a fixed parameter ``trial``.
        ``train_builder`` (``Callable``): The function to build a training configuration with two fixed parameters ``trial`` and ``model``.
        ``evaluator`` (``dhg.metrics.BaseEvaluator``): The DHG evaluator object to evaluate performance of the model in the experiment.
        ``device`` (``torch.device``): The target device to run the experiment.
        ``structure_builder`` (``Optional[Callable]``): The function to build a structure with a fixed parameter ``trial``. The structure can be ``dhg.Graph``, ``dhg.DiGraph``, ``dhg.BiGraph``, and ``dhg.Hypergraph``.
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
        structure_builder: Optional[Callable] = None,
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
            structure_builder=structure_builder,
            study_name=study_name,
            overwrite=overwrite,
        )
        self.to(self.device)

    def to(self, device: torch.device):
        r"""Move the input data to the target device.

        Args:
            ``device`` (``torch.device``): The specified target device to store the input data.
        """
        self.device = device
        for name in self.vars_for_DL:
            if name in self.data.keys():
                self.data[name] = self.data[name].to(device)
        return self

    @property
    def vars_for_DL(self):
        r"""Return a name list for available variables for deep learning in the vertex classification task. The name list includes ``features``, ``structure``, ``labels``, ``train_mask``, ``val_mask``, and ``test_mask``.
        """
        return (
            "features",
            "structure",
            "labels",
            "train_mask",
            "val_mask",
            "test_mask",
        )

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
        features, structure = data["features"], data["structure"]
        train_mask, labels = data["train_mask"], data["labels"]
        model.train()
        optimizer.zero_grad()
        outputs = model(features, structure)
        loss = criterion(outputs[train_mask], labels[train_mask],)
        loss.backward()
        optimizer.step()

    @torch.no_grad()
    def validate(self, data: dict, model: nn.Module):
        r"""Validate the model.

        Args:
            ``data`` (``dict``): The input data.
            ``model`` (``nn.Module``): The model.
        """
        features, structure = data["features"], data["structure"]
        val_mask, labels = data["val_mask"], data["labels"]
        model.eval()
        outputs = model(features, structure)
        res = self.evaluator.validate(labels[val_mask], outputs[val_mask])
        return res

    @torch.no_grad()
    def test(self, data: Optional[dict] = None, model: Optional[nn.Module] = None):
        r"""Test the model.

        Args:
            ``data`` (``dict``, optional): The input data if set to ``None``, the specified ``data`` in the intialization of the experiments will be used. Defaults to ``None``.
            ``model`` (``nn.Module``, optional): The model if set to ``None``, the trained best model will be used. Defaults to ``None``.
        """
        if data is None:
            features, structure = self.data["features"], self.best_structure
            test_mask, labels = self.data["test_mask"], self.data["labels"]
        else:
            features, structure = (
                data["features"].to(self.device),
                data["structure"].to(self.device),
            )
            test_mask, labels = (
                data["test_mask"].to(self.device),
                data["labels"].to(self.device),
            )
        if model is None:
            model = self.best_model
        model = model.to(self.device)
        model.eval()
        outputs = model(features, structure)
        res = self.evaluator.test(labels[test_mask], outputs[test_mask])
        return res
