import os
import random
import pytest

import torch

from code.dataset import PointCloudEmbeddingDataset, PointCloudEmbeddingSequenceDataset
from models.DeepCAD.config.configAE import ConfigAE
from models.DeepCAD.trainer.trainerAE import TrainerAE

DATA_DIR = "data"

@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_PointCloudEmbeddingDataset_empty(split):
    """Check if the datasets are not empty."""
    dataset = PointCloudEmbeddingDataset(DATA_DIR, split)
    assert len(dataset) > 0, f"Dataset {split} is empty."
@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_PointCloudEmbeddingSequenceDataset_empty(split):
    """Check if the datasets are not empty."""
    dataset = PointCloudEmbeddingSequenceDataset(DATA_DIR, split)
    assert len(dataset) > 0, f"Dataset {split} is empty."

@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_PointCloudEmbeddingDataset_pc_cad_seq_match(split):
    """Check whether the point cloud path and CAD-sequence path match"""
    dataset = PointCloudEmbeddingDataset(DATA_DIR, split)

    idx = random.randint(0, len(dataset) - 1)
    pc_path = dataset.get_pc_path(idx)
    cad_path = dataset.get_cad_seq_path(idx)

    assert (
        os.path.splitext(os.path.basename(pc_path))[0] ==
        os.path.splitext(os.path.basename(cad_path))[0]
    ), "Point Cloud and CAD-Sequence don't align in PointCloudEmbeddingDataset {split} set."
@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_PointCloudEmbeddingSequenceDataset_pc_cad_seq_match(split):
    """Check whether the point cloud path and CAD-sequence path match"""
    dataset = PointCloudEmbeddingSequenceDataset(DATA_DIR, split)

    idx = random.randint(0, len(dataset) - 1)
    pc_path = dataset.get_pc_path(idx)
    cad_path = dataset.get_cad_seq_path(idx)

    assert (
        os.path.splitext(os.path.basename(pc_path))[0] ==
        os.path.splitext(os.path.basename(cad_path))[0]
    ), "Point Cloud and CAD-Sequence don't align in PointCloudEmbeddingSequenceDataset {split} set."


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_cad_vec_lat_rep_match(split):
    """ Test whether the target latent representations of the point cloud sampled 
        from its CAD-sequence in PN++ training matches the encoded CAD-sequences 
        by DeepCAD.
        ATTENTION: If the pretrained DeepCAD model specified in ConfigAE changes,
        the latent representations have to be encoded again by the new DeepCAD model.
        """ 
    
    dataset = PointCloudEmbeddingSequenceDataset(DATA_DIR, split)
    idx = random.randint(0, len(dataset) - 1)
    data = dataset[idx]
    lat_rep, cad_vec = data['z'], data['tgt_vec']
    command = cad_vec[:, 0]
    args = cad_vec[:, 1:]
    cad_vec = {"command": command, "args": args}

    cfg = ConfigAE('test', parse=False) # parse=True would consume pytest's argv
    tr_agent = TrainerAE(cfg)
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    with torch.no_grad():
        output = tr_agent.encode(cad_vec)
        output = output.squeeze()
        assert(torch.allclose(lat_rep, output, atol=1e-6)), "The encoded CAD-\
            sequence does not match the latent representation on disk."
