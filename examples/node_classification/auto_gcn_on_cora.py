import torch
import torch.nn as nn
import torch.optim as optim

from dhg import Graph
from dhg.data import Cora
from dhg.models import GCN
from dhg.random import set_seed
from dhg.experiments import GraphVertexClassificationTask as Task
from dhg.metrics import GraphVertexClassificationEvaluator as Evaluator


def model_builder(trial):
    return GCN(ft_dim, trial.suggest_int("hidden_dim", 8, 32), num_classes)


def train_builder(trial, model):
    optimizer = optim.Adam(model.parameters(), lr=trial.suggest_loguniform("lr", 1e-4, 1e-2), weight_decay=5e-4,)
    criterion = nn.CrossEntropyLoss()
    return {
        "optimizer": optimizer,
        "criterion": criterion,
    }


if __name__ == "__main__":
    work_root = "/home/fengyifan/OS3D/toolbox/exp_cache/tmp"
    set_seed(2022)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data = Cora()
    num_v, ft_dim = data["features"].shape
    num_classes = data["labels"].max().item() + 1
    input_data = {
        "features": data["features"],
        "structure": Graph(num_v, data["edge_list"]),
        "labels": data["labels"],
        "train_mask": data["train_mask"],
        "val_mask": data["val_mask"],
        "test_mask": data["test_mask"],
    }
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    task = Task(work_root, input_data, model_builder, train_builder, evaluator, device,)
    task.run(200, 50, "maximize")
