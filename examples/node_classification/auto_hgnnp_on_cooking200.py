import torch
import torch.nn as nn
import torch.optim as optim

from dhg import Graph, Hypergraph
from dhg.data import Cooking200
from dhg.models import HGNN, HGNNP
from dhg.random import set_seed
from dhg.experiments import HypergraphVertexClassificationTask as Task
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator


def structure_builder(trial):
    global hg_base, g
    cur_hg: Hypergraph = hg_base.clone()
    return cur_hg


def model_builder(trial):
    return HGNNP(dim_features, trial.suggest_int("hidden_dim", 10, 20), num_classes, use_bn=True)


def train_builder(trial, model):
    optimizer = optim.Adam(
        model.parameters(),
        lr=trial.suggest_loguniform("lr", 1e-4, 1e-2),
        weight_decay=trial.suggest_loguniform("weight_decay", 1e-4, 1e-2),
    )
    criterion = nn.CrossEntropyLoss()
    return {
        "optimizer": optimizer,
        "criterion": criterion,
    }


if __name__ == "__main__":
    work_root = "/home/fengyifan/OS3D/toolbox/exp_cache/tmp"
    set_seed(2022)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data = Cooking200()
    dim_features = data["num_vertices"]
    num_classes = data["num_classes"]
    hg_base = Hypergraph(data["num_vertices"], data["edge_list"])
    input_data = {
        "features": torch.eye(data["num_vertices"]),
        "labels": data["labels"],
        "train_mask": data["train_mask"],
        "val_mask": data["val_mask"],
        "test_mask": data["test_mask"],
    }
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    task = Task(
        work_root, input_data, model_builder, train_builder, evaluator, device, structure_builder=structure_builder,
    )
    task.run(200, 50, "maximize")
