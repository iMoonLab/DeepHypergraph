import abc
import time
import shutil
import logging
from pathlib import Path
from copy import deepcopy
from typing import Callable, Optional, Callable, Union

import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler

import dhg
from dhg.metrics import BaseEvaluator
from dhg.utils import default_log_formatter
from dhg.structure.base import load_structure


class BaseTask:
    r"""The base class of Auto-experiment in DHG.

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
        self.data = data
        self.model_builder = model_builder
        self.train_builder = train_builder
        self.structure_builder = structure_builder
        self.evaluator = evaluator
        self.device = device
        self.study = None
        if study_name is None:
            self.study_name = time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
        else:
            self.study_name = study_name
        work_root = Path(work_root)
        self.study_root = work_root / self.study_name
        if overwrite and self.study_root.exists():
            shutil.rmtree(self.study_root)
        self.log_file = self.study_root / "log.txt"
        self.cache_root = self.study_root / "cache"
        if not work_root.exists():
            if work_root.parent.exists():
                work_root.mkdir(exist_ok=True)
            else:
                raise ValueError(f"The work_root {work_root} does not exist.")
        self.study_root.mkdir(exist_ok=True)
        self.cache_root.mkdir(exist_ok=True)
        # configure logging
        self.logger = optuna.logging.get_logger("optuna")
        self.logger.setLevel(logging.INFO)
        out_file_handler = logging.FileHandler(self.log_file, mode="a", encoding="utf8")
        out_file_handler.setFormatter(default_log_formatter())
        self.logger.addHandler(out_file_handler)
        self.logger.info(f"Logs will be saved to {self.log_file.absolute()}")
        self.logger.info(f"Files in training will be saved in {self.study_root.absolute()}")

    def experiment(self, trial: optuna.Trial):
        r"""Run the experiment for a given trial.

        Args:
            ``trial`` (``optuna.Trial``): The ``optuna.Trial`` object.
        """
        if self.structure_builder is not None:
            self.data["structure"] = self.structure_builder(trial).to(self.device)
        model = self.model_builder(trial).to(self.device)
        train_configs: dict = self.train_builder(trial, model)
        assert "optimizer" in train_configs.keys()
        optimizer = train_configs["optimizer"]
        assert "criterion" in train_configs.keys()
        criterion = train_configs["criterion"]
        scheduler = train_configs.get("scheduler", None)

        best_model = None
        if self.direction == "maximize":
            best_score = -float("inf")
        else:
            best_score = float("inf")
        for epoch in range(self.max_epoch):
            self.train(self.data, model, optimizer, criterion)
            val_res = self.validate(self.data, model)
            trial.report(val_res, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if scheduler is not None:
                scheduler.step()
            if self.direction == "maximize":
                if val_res > best_score:
                    best_score = val_res
                    best_model = deepcopy(model)
        with open(self.cache_root / f"{trial.number}_model.pth", "wb") as f:
            torch.save(best_model.cpu().state_dict(), f)
        self.data["structure"].save(self.cache_root / f"{trial.number}_structure.dhg")
        return best_score

    def _remove_cached_data(self):
        r"""Remove cached models and structures.
        """
        if self.study is not None:
            for filename in self.cache_root.glob("*"):
                if filename.stem.split("_")[0] != str(self.study.best_trial.number):
                    filename.unlink()

    def run(self, max_epoch: int, num_trials: int = 1, direction: str = "maximize"):
        r"""Run experiments with automatically hyper-parameter tuning.

        Args:
            ``max_epoch`` (``int``): The maximum number of epochs to train for each experiment.
            ``num_trials`` (``int``): The number of trials to run. Defaults to ``1``.
            ``direction`` (``str``): The direction to optimize. Defaults to ``"maximize"``.
        """
        self.logger.info(f"Random seed is {dhg.random.seed()}")
        sampler = TPESampler(seed=dhg.random.seed())
        self.max_epoch, self.direction = max_epoch, direction
        self.study = optuna.create_study(direction=direction, sampler=sampler)
        self.study.optimize(self.experiment, n_trials=num_trials, timeout=600)

        self._remove_cached_data()
        self.best_model = self.model_builder(self.study.best_trial)
        self.best_model.load_state_dict(torch.load(f"{self.cache_root}/{self.study.best_trial.number}_model.pth"))
        self.best_structure = load_structure(f"{self.cache_root}/{self.study.best_trial.number}_structure.dhg")
        self.best_model = self.best_model.to(self.device)
        self.best_structure = self.best_structure.to(self.device)

        self.logger.info("Best trial:")
        self.best_trial = self.study.best_trial
        self.logger.info(f"\tValue: {self.best_trial.value:.3f}")
        self.logger.info(f"\tParams:")
        for key, value in self.best_trial.params.items():
            self.logger.info(f"\t\t{key} |-> {value}")
        test_res = self.test()
        self.logger.info(f"Final test results:")
        for key, value in test_res.items():
            self.logger.info(f"\t{key} |-> {value:.3f}")

    @abc.abstractmethod
    def train(
        self, data: dict, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module,
    ):
        r"""Train model for one epoch.

        Args:
            ``data`` (``dict``): The input data.
            ``model`` (``nn.Module``): The model.
            ``optimizer`` (``torch.optim.Optimizer``): The model optimizer.
            ``criterion`` (``nn.Module``): The loss function.
        """

    @torch.no_grad()
    @abc.abstractmethod
    def validate(
        self, data: dict, model: nn.Module,
    ):
        r"""Validate the model.

        Args:
            ``data`` (``dict``): The input data.
            ``model`` (``nn.Module``): The model.
        """

    @torch.no_grad()
    @abc.abstractmethod
    def test(self, data: Optional[dict] = None, model: Optional[nn.Module] = None):
        r"""Test the model.

        Args:
            ``data`` (``dict``, optional): The input data if set to ``None``, the specified ``data`` in the intialization of the experiments will be used. Defaults to ``None``.
            ``model`` (``nn.Module``, optional): The model if set to ``None``, the trained best model will be used. Defaults to ``None``.
        """
