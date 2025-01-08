Extract the downloaded DeepCAD dataset into this directory. The data should contain the following directories:
- `cad_json` contains the original json files that we parsed from Onshape and each file describes a CAD construction sequence.
- `cad_vec` contains our vectorized representation for CAD sequences, which serves for fast data loading. They can also be obtained using `dataset/json2vec.py`.
- `train_val_test_split.json` contains the train - val - test split.
