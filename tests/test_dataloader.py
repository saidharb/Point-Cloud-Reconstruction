import os
import random
import pytest

from code.dataset import PointCloudEmbeddingDataset


# Check if the Point Cloud and latent representation provided by PointCloudEmbeddingDataset match
DATA_DIR = "data"

# Check if PC and CAD-sequence paths match in PN++ training
@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_pc_lat_rep_match(split):
    dataset = PointCloudEmbeddingDataset(DATA_DIR, split)

    assert len(dataset) > 0, f"Dataset {split} is empty."

    idx = random.randint(0, len(dataset) - 1)
    pc_path = dataset.get_pc_path(idx)
    cad_path = dataset.get_cad_seq_path(idx)

    assert (
        os.path.splitext(os.path.basename(pc_path))[0] ==
        os.path.splitext(os.path.basename(cad_path))[0]
    ), "Point Cloud and CAD-Sequence don't align in PointCloudEmbeddingDataset {split} set."


# schauen ob pretrained DeepCAD model die selbe lat rep produziert
# die als target hier is, 
# Input: CAD-seq (same as PC)
# Infer CAD-seq using pretrained DeepCAD
# compare if DeepCAD pred lat rep is equal to the target