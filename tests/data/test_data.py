import torch
import pytest
from dhg.data import Cora, Citeseer, Pubmed, Cooking200


def test_cora():
    data = Cora()
    data["features"]
    assert data.content == [
        "num_classes",
        "num_vertices",
        "num_edges",
        "dim_features",
        "features",
        "edge_list",
        "labels",
        "train_mask",
        "val_mask",
        "test_mask",
    ]
    assert len(data["edge_list"]) == 10858
    assert data["features"].shape == (2708, 1433)
    assert data["labels"].dtype == torch.long
    assert data["train_mask"].dtype == torch.bool
    assert data["train_mask"].sum() == 140


def test_citeseer():
    data = Citeseer()
    data["features"]
    assert data.content == [
        "num_classes",
        "num_vertices",
        "num_edges",
        "dim_features",
        "features",
        "edge_list",
        "labels",
        "train_mask",
        "val_mask",
        "test_mask",
    ]
    assert len(data["edge_list"]) == 9464
    assert data["features"].shape == (3327, 3703)
    assert data["labels"].dtype == torch.long
    assert data["train_mask"].dtype == torch.bool
    assert data["train_mask"].sum() == 120


def test_pubmed():
    data = Pubmed()
    data["features"]
    assert data.content == [
        "num_classes",
        "num_vertices",
        "num_edges",
        "dim_features",
        "features",
        "edge_list",
        "labels",
        "train_mask",
        "val_mask",
        "test_mask",
    ]
    assert len(data["edge_list"]) == 88676
    assert data["features"].shape == (19717, 500)
    assert data["labels"].dtype == torch.long
    assert data["train_mask"].dtype == torch.bool
    assert data["train_mask"].sum() == 60


def test_cooking():
    data = Cooking200()
    assert data.content == [
        "num_classes",
        "num_vertices",
        "num_edges",
        "edge_list",
        "labels",
        "train_mask",
        "val_mask",
        "test_mask",
    ]
    assert len(data["edge_list"]) == 2755
    assert data["labels"].dtype == torch.long
    assert data["train_mask"].dtype == torch.bool
    assert data["train_mask"].sum() == 200
