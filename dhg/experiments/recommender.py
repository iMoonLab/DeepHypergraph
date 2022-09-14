from typing import Callable, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
import optuna

from dhg.metrics import BaseEvaluator
from .base import BaseTask


class UserItemRecommenderTask(BaseTask):
    r"""The auto-experiment class for the recommender task on user-item bipartite graph.

    Args:
        ``work_root`` (``Optional[Union[str, Path]]``): User's work root to store all studies.
        ``data`` (``dict``): The dictionary to store input data that used in the experiment.
        ``model_builder`` (``Callable``): The function to build a model with a fixed parameter ``trial``.
        ``train_builder`` (``Callable``): The function to build a training configuration with two fixed parameters ``trial`` and ``model``.
        ``evaluator`` (``dhg.metrics.BaseEvaluator``): The DHG evaluator object to evaluate performance of the model in the experiment.
        ``device`` (``torch.device``): The target device to run the experiment.
        ``structure_builder`` (``Optional[Callable]``): The function to build a structure with a fixed parameter ``trial``. The structure should be ``dhg.DiGraph``.
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
        r"""Return a name list for available deep learning variables for the recommender task on user-item bipartite graph. The name list includes ``structure``.
        """
        return ("structure",)

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
        structure = data.get("structure", None)
        train_loader = data["train_loader"]
        model.train()
        for users, pos_items, neg_items in train_loader:
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)
            optimizer.zero_grad()
            if structure is not None:
                emb_u, emb_i = model(structure)
            else:
                emb_u, emb_i = model()
            loss = criterion(emb_u, emb_i, users, pos_items, neg_items, model=model)
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def validate(self, data: dict, model: nn.Module):
        r"""Validate the model.

        Args:
            ``data`` (``dict``): The input data.
            ``model`` (``nn.Module``): The model.
        """
        structure = data.get("structure", None)
        test_loader = data["test_loader"]
        model.eval()
        for users, train_mask, true_rating in test_loader:
            users = users.to(self.device)
            train_mask = train_mask.to(self.device)
            true_rating = true_rating.to(self.device)
            if structure is not None:
                emb_u, emb_i = model(structure)
            else:
                emb_u, emb_i = model()
            pred_rating = torch.mm(emb_u[users], emb_i.t())
            pred_rating = pred_rating + train_mask
            self.evaluator.validate_add_batch(true_rating, pred_rating)
        res = self.evaluator.validate_epoch_res()
        return res

    @torch.no_grad()
    def test(self, data: Optional[dict] = None, model: Optional[nn.Module] = None):
        r"""Test the model.

        Args:
            ``data`` (``dict``, optional): The input data if set to ``None``, the specified ``data`` in the intialization of the experiments will be used. Defaults to ``None``.
            ``model`` (``nn.Module``, optional): The model if set to ``None``, the trained best model will be used. Defaults to ``None``.
        """
        if data is None:
            structure = self.best_structure
            test_loader = self.data["test_loader"]
        else:
            structure = data.get("structure", None)
            test_loader = data["test_loader"]
        if structure is not None:
            structure = structure.to(self.device)
        if model is None:
            model = self.best_model
        model = model.to(self.device)
        model.eval()
        for users, train_mask, true_rating in test_loader:
            users = users.to(self.device)
            true_rating = true_rating.to(self.device)
            if structure is not None:
                emb_u, emb_i = model(structure)
            else:
                emb_u, emb_i = model()
            pred_rating = torch.mm(emb_u[users], emb_i.t())
            pred_rating = pred_rating + train_mask
            self.evaluator.test_add_batch(true_rating, pred_rating)
        res = self.evaluator.test_epoch_res()
        return res
