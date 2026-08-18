# Extrusion Segmentation Strategy to improve CAD Reconstruction from Point Cloud

Code and dataset tooling for:

> Harb, S., Maboudi, M., and Gerke, M.: **Extrusion Segmentation Strategy to improve CAD
> Reconstruction from Point Cloud**, *The International Archives of the Photogrammetry, Remote
> Sensing and Spatial Information Sciences*, XLIX-B2-2026, 189-197, 2026.
> [doi:10.5194/isprs-archives-XLIX-B2-2026-189-2026](https://doi.org/10.5194/isprs-archives-XLIX-B2-2026-189-2026)

Developed at the Institute of Geodesy and Photogrammetry (IGP), TU Braunschweig.

The pipeline predicts CAD command sequences from point clouds: PointNet++ encodes a point cloud
into a 256-dimensional latent space, and the DeepCAD decoder decodes the latent vector into a
CAD sequence. **The paper describes the method, the segmentation strategy and the results.**
This README covers only what the paper does not: how to obtain the data, how to regenerate the
derived dataset, and how to run each script.

## Repository layout

| Path | Contents |
|---|---|
| `code/` | Training, testing and dataset-generation scripts for this project |
| `models/DeepCAD/` | Vendored [DeepCAD](https://github.com/ChrisWu1997/DeepCAD), modified (see its `CHANGELOG.md`) |
| `models/Pointnet_Pointnet2_pytorch/` | Vendored [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), modified (see its `CHANGELOG.md`) |
| `notebooks/` | Exploratory notebooks used during development |
| `tests/` | Dataset consistency checks, run with `pytest` |
| `utils/` | Standalone helper scripts |
| `data/` | Datasets (not tracked; see [Data](#data)) |
| `check_extrusion_pairing.py`, `verify_extrusion_pairing.py` | Dataset validation, see [Validating the dataset](#validating-the-dataset) |
| `pc2cad.py` | Full point cloud to CAD inference pipeline |
| `*.job` | SLURM submission scripts used for the cluster runs |

## DeepCAD
The DeepCAD model from the [DeepCAD: A Deep Generative Network for Computer-Aided Design Models](https://arxiv.org/abs/2105.09492) ICCV 2021 by Wu et al. serves two purposes: first as an encoder for CAD sequences and second as the decoder for the point cloud latent representations. For the purpose of this project the original code was modified (cf. [CHANGELOG.md](https://github.com/saidharb/Point-Cloud-Reconstruction/blob/main/models/DeepCAD/CHANGELOG.md)).

### Data
First you can download the data from the [DeepCAD dataset](http://www.cs.columbia.edu/cg/deepcad/data.tar) and extract it in the `data` folder in the main directory.

The derived extrusion-segmentation data described in [Extrusion Segmentation Dataset](#extrusion-segmentation-dataset) (`pc_from_vec`, `pc_from_vec_labels`, `pc_extrusion`, `pc_extrusion_labels`) is published separately so it does not have to be regenerated: TODO(DOI) — add the data repository link and DOI here once it is minted.

### Turn CAD Sequences into Point Clouds
To convert CAD sequences from the DeepCAD dataset into point clouds, use the `json2pc.py` script:
```bash
$ cd models/DeepCAD/dataset
$ python json2pc.py
```
This will create a new `pc_cad` directory within the `data` directory, containing the `.ply` point cloud files for the CAD sequences in the DeepCAD dataset.

### Encode CAD Sequences into Latent Representation
#### Download pretrained model
You can use a pretrained DeepCAD model from the original authors under this [link](http://www.cs.columbia.edu/cg/deepcad/pretrained.tar). Extract it in the `data` directory into the `latent` directory.
#### Encode the CAD sequences
From the main repository execute the following commands to encode the CAD sequences into the latent representation:
```bash
$ cd models/DeepCAD
$ python test.py --proj_dir "../../data/latent" --exp_name "pretrained" --mode "enc" --ckpt 1000 --data_root "../../data"  
```
The pretrained model will encode the CAD sequences and save the results in the `data/latent/pretrained/results` directory.

## Extrusion Segmentation Dataset
The baseline pipeline maps a whole point cloud to a whole CAD sequence (the *monolithic* model, MEM). The extrusion segmentation model (SEM) instead splits a point cloud into one cloud per extrusion and predicts a short CAD subsequence for each. The data for this is derived from the DeepCAD dataset in two steps.

```
cad_json ──json2pc.py──> pc_cad                                       (baseline / MEM)

cad_json ──DeepCAD──> cad_vec ──code/create_data.py──> pc_from_vec
                                                       pc_from_vec_labels
                                  └──code/create_extr_data.py──> pc_extrusion
                                                                 pc_extrusion_labels  (SEM)
```

The second script consumes the output of the first, so they must be run **in this order**. Both are run from the repository root and take a `--data-root` argument (default `data`, resolved relative to the current working directory, matching `pc2cad.py` and `code/train.py`):

```bash
$ export PYTHONPATH=$(pwd):$PYTHONPATH
$ python code/create_data.py --data-root data
$ python code/create_extr_data.py --data-root data
```

Both scripts write into `--data-root` and overwrite whatever is already there, so point them at a scratch directory if you want to keep the released data intact.

Each script prints a running failure count while it works and a `N of M samples failed` summary with the most common exception types at the end. Individual samples can legitimately fail on malformed CAD, so the broad exception handler stays, but a failure rate above 1% is treated as systematic and the script exits non-zero.

### `code/create_data.py`
Rebuilds each model from its `cad_vec` sequence with the OpenCASCADE kernel, samples a ground-truth cloud from the whole solid and one cloud per extrusion, keeps only the per-extrusion points that lie within `epsilon = 0.005` of a ground-truth point, and resamples the survivors to a fixed 10,000 points. It also repairs extrusion commands whose numericalised distance or scale collapsed to zero, writing the corrected sequence back into `cad_vec`.

Outputs:

| Path | Contents |
|---|---|
| `pc_from_vec/<bucket>/<id>.ply` | 10,000 points sampled from the whole model |
| `pc_from_vec_labels/<bucket>/<id>.h5` | per-point extrusion labels and the padded subsequence of every extrusion |

`pc_from_vec_labels/<bucket>/<id>.h5`:

| Dataset | Shape | dtype | Meaning |
|---|---|---|---|
| `labels` | `(10000,)` | `int64` | extrusion id of each point in the matching `.ply`, in file order |
| `sequences/<k>` | `(60, 17)` | `int64` | the CAD subsequence of extrusion `k`, EOS-padded to the full length; `k` runs over `0 .. n_extrusions-1` |

**Note that `labels` need not contain every `k`.** If the proximity filter removes all points of an extrusion, that extrusion has no label in `labels` while its subsequence is still stored under `sequences/<k>`.

### `code/create_extr_data.py`
Partitions each `pc_from_vec` cloud by its labels and writes one point cloud plus one CAD subsequence per surviving extrusion. It needs all 10,000 points and states that sample size explicitly, and it seeds the point subsample so a regeneration is deterministic.

Outputs:

| Path | Contents |
|---|---|
| `pc_extrusion/<bucket>/<id>/<id>_<j>.ply` | the points of the `j`-th surviving extrusion |
| `pc_extrusion_labels/<bucket>/<id>/<id>_<j>.h5` | that extrusion's id and CAD subsequence |

`pc_extrusion_labels/<bucket>/<id>/<id>_<j>.h5`:

| Dataset | Shape | dtype | Meaning |
|---|---|---|---|
| `extrusion_id` | scalar | `int64` | the extrusion this cloud was sampled from |
| `sequence` | `(N, 17)` | `int64` | that extrusion's CAD subsequence, truncated at and including the first EOS |

**`j` is a file index, not an extrusion id.** The filenames are always contiguous (`_0`, `_1`, `_2`, …) so that the paths of the published data stay stable, but if an extrusion lost all of its points the ids skip a value. The authoritative id is the `extrusion_id` dataset inside the `.h5`: a model whose labels are `[0, 1, 3, 4, 6]` produces files `_0 .. _4` carrying `extrusion_id` `0, 1, 3, 4, 6`.

### Validating the dataset
Two scripts in the repository root check the extrusion pairing. **They answer different questions and are not interchangeable.**

`check_extrusion_pairing.py` **characterises the source data.** It reads `pc_from_vec_labels/` and reports which models lost an extrusion to the proximity filter, and how many samples the enumeration-index bug described in [Known issues and corrections](#known-issues-and-corrections) would therefore affect. Those numbers are a property of `pc_from_vec_labels` and are **identical before and after any fix** — a non-zero "MISPAIRED" count here does *not* mean the generated data is wrong. This is not a post-fix verifier.

```bash
$ python check_extrusion_pairing.py --data-root data
==============================================================
models scanned                        177,776
models with a dropped extrusion           587
  - dropped at the tail (harmless)        162
  - dropped mid-sequence (BAD)            425
extrusions with no surviving points     1,015
MISPAIRED point cloud / sequence        1,137
==============================================================
```

This is the expected output for the released data: 425 models and 1,137 samples *would have been* mispaired by the original generator, and the script reports the same figures after the correction because it never looks at the generated data.

`verify_extrusion_pairing.py` **verifies the generated data.** It reads `pc_extrusion_labels/` and compares every `extrusion_id` and `sequence` on disk against the subsequence the cloud should be paired with. It exits non-zero on any mismatch, so it is the one to run after a regeneration or before a release. `--check-clouds` additionally compares the `.ply` point counts against the label groups, which confirms the cloud and the sequence describe the same extrusion.

```bash
$ python verify_extrusion_pairing.py --data-root data
==============================================================
models verified                       177,776
samples verified                      362,225
MISMATCHED point cloud / sequence           0
==============================================================
all point clouds are paired with the correct CAD subsequence.
$ echo $?
0
```

This is the expected output for the released data. A full pass takes roughly half an hour.

Both take `--report <file.csv>` to dump their findings.

## PointNet++
Until this point, the CAD-sequence dataset is downloaded, the CAD-sequences are converted to point clouds and the CAD-sequences are encoded into the latent space by the DeepCAD model. The next step is to train the PointNet++ model in a regression task on the point clouds with the latent representation of the CAD-sequences as targets. PointNet++ is from the paper [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413) NIPS 2017 by Qi et al. and for this project the [PyTorch version](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) was used and modified (cf. [CHANGELOG.md](https://github.com/saidharb/Point-Cloud-Reconstruction/blob/main/models/Pointnet_Pointnet2_pytorch/CHANGELOG.md)). 

### Setup
The project targets **Python 3.9**. Create the conda environment from `environment.yml`, which
pins the interpreter and `pythonocc-core` (the OpenCASCADE binding that `code/create_data.py`
requires and that pip cannot provide):

```bash
$ conda env create -f environment.yml
$ conda activate pointnet_venv
$ pip install -r requirements.txt
```

If pip cannot find an appropriate version of open3d, install it from conda-forge:
```bash
$ conda install -c conda-forge open3d
```

PointNet++ needs its CUDA extensions compiled; follow the build instructions of the vendored
[Pointnet_Pointnet2_pytorch](models/Pointnet_Pointnet2_pytorch/README.md) for a GPU setup.

### Train PointNet++
In order to train PointNet++ a training script is developed, which will dynamically adapt to CPU or GPU. Before executing the training script, two environment variables have to be set, in particular the `PYTHONPATH` and the `WANDB_API_KEY`, if training should be tracked using Weight and Biases. The API key can be obtained from your user account at WandB. Set the environment variables from the root repository like this:

```bash
$ export PYTHONPATH=$(pwd):$PYTHONPATH
$ export WANDB_API_KEY=<WandB API key>
```
**Note**: Always set the PYTHONPATH variable when starting a new session.

Then the training script can be executed from the root repository. Note the command line arguments listed below to modify training.
```bash
$ python code/train.py
```
| Argument           | Type    | Default      | Description                                                                 |
|--------------------|---------|--------------|-----------------------------------------------------------------------------|
| `--data_root`      | `str`   | `'data'`     | Data directory relative to root directory                                   |
| `--batch_size`     | `int`   | `24`         | Batch size                                                                 |
| `--max_epochs`     | `int`   | `50`         | Maximum number of epochs                                                   |
| `--save_interval`  | `int`   | `20`         | Save interval for models                                                   |
| `--learning_rate`  | `float` | `0.001`      | Initial learning rate                                                      |
| `--lr_patience`    | `int`   | `15`         | Patience in epochs for learning rate decay                                 |
| `--early_stopping` | `int`   | `20`         | Abort training after this number of epochs with no validation loss decrease|
| `--verbose`        | `bool`  | `False`      | Output per batch metrics                                                   |
| `--wandb`          | `bool`  | `False`      | Enable WandB tracking                                                      |
| `--name`           | `str`   | `'test_run'` | Name of WandB run                                                          |
| `--output_dir`     | `str`   | required     | Name of the model save directory relative to root directory             |
| `--lr_type`     | `str`   | `'step'`     | Learning rate schedule, one of `step`, `step_adv` or `cosine`: `step` for a simple step learning rate scheduler, `step_adv` for reducing learning rate on val_loss plateau or `cosine` for cosine annealing with warm restarts|
| `--msg`     | `bool`   | `False`     | Use multi-scale-grouping instead of single-scale-grouping       |
| `--aug`     | `bool`   | `False`     | Turn on point cloud augmentation (random dropout, scale and shift)       |
                        

The training script will create a directory `models/trained_models` where all training runs are saved including the best and last model, checkpoint models, logging information and a csv file containing the metrics. The learning rate is halved if the validation loss did not decrease for the number of epochs specified in `lr_patience`.

### Test PointNet++
You can test the performance of PointNet++ to encode point clouds into the same latent space as the CAD-sequences using the `code/test.py` script:
```bash
$ python code/test.py --model_path <path/to/model/from/root>
```
The script will create a `test.log` and a `test_metrics_<model_name>.csv` file within the particular models directory saving the test metrics (MSE, RMSE, MAE).
| Argument           | Type    | Default      | Description                                                                 |
|--------------------|---------|--------------|-----------------------------------------------------------------------------|
| `--data_root`      | `str`   | `'data'`     | Data directory relative to root directory                                   |
| `--model_path`     | `str`   | required     | Path to the trained model                                                 |
| `--batch_size`     | `int`   | `24`         | Batch size                                                                 |
| `--verbose`        | `bool`  | `False`      | Output per batch metrics                                                   |

## Extrusion Segmentation Models
The scripts below implement the monolithic (MEM) and extrusion segmentation (SEM) models that
the paper compares. **See the paper for the strategy, the model variants and the results** — this
section only records which script consumes which dataset, so the right one can be found.

All training scripts take the same arguments as [`code/train.py`](#train-pointnet) and all testing
scripts the same as [`code/test.py`](#test-pointnet), plus `--gpu`.

### Training

| Script | Input | Target |
|---|---|---|
| `code/train_pn2_ext_seg.py` | `pc_from_vec` cloud | per-point extrusion label (the segmentation stage) |
| `code/train_primitive_extr.py` | `pc_extrusion` cloud of one extrusion | that extrusion's CAD subsequence |
| `code/train_complex_extr.py` | `pc_from_vec` cloud of a whole model | the model's full CAD sequence |

```bash
$ python code/train_pn2_ext_seg.py --output_dir "partseg_run" --gpu --verbose
```

### Testing
The test scripts are named `test_<model>_on_<evaluation set>`, where *simple* models consist of a
single extrusion and *complex* models of several. Each writes `test_metrics.csv`,
`test_metrics.npz` and `test_per_sl_metrics.pkl` into the model directory.

| Script | Model | Evaluated on |
|---|---|---|
| `code/test_pn2_ext_seg.py` | segmentation network | extrusion segmentation accuracy |
| `code/test_primitive_extr.py` | SEM | single-extrusion clouds |
| `code/test_complex_extr.py` | MEM | simple and complex models |
| `code/test_primitive_on_single_extr.py` | SEM | simple models |
| `code/test_primitive_on_complex.py` | SEM | simple and complex models, by inferring primitives |
| `code/test_complex_on_single_extr.py` | MEM | simple models |
| `code/test_two_stage_on_single_extr.py` | two-stage baseline | simple models |
| `code/test_two_stage_on_complex.py` | two-stage baseline | simple and complex models |
| `code/test_MEM_on_complex.py` | MEM | complex models, writes `test_results_MEM_complex.pkl` |
| `code/test_SEM_on_complex.py` | SEM | complex models, writes `test_results_SEM_complex.pkl` |

The last two produce the MEM/SEM comparison reported in the paper.

## PC to CAD Pipeline
The script `pc2cad.py` enables the full inference pipeline using PointNet++ as the encoder for point clouds and the DeepCAD decoder to decode them into CAD-sequences. Previously DeepCAD was trained on auto-encoding CAD-sequences and PointNet++ was trained on encoding point clouds into the same latent space as DeepCAD. The pre-trained DeepCAD model should be within `data/latent/pretrained/model/ckpt_epoch1000.pth`. During inference metrics like the PointNet++ MSE-Loss and the DeepCAD Command- and Argument-Loss are recorded and predicted latent representation and CAD-sequence are saved within the model directory under `results`. Run the script using the following command from root:

```bash
$ python pc2cad.py --exp_name "test_experiment" --model_path "path/to/model.pth"
```

| Argument      | Type   | Default  | Description  |
|--------------|--------|----------|-------------|
| `--data_root` | `str`  | `'data'`  | Data directory relative to root directory |
| `--batch_size` | `int`  | `48`  | Batch size |
| `--verbose` | `bool`  | `False` | Output per batch metrics |
| `--exp_name` | `str`  | **Required** | Name of the experiment in the results folder within the run directory |
| `--model_path` | `str`  | **Required** | Path to the trained PointNet++ model |
| `--save` | `bool` | `False` | Save predicted latent representations and CAD sequences |
| `--phase` | `str`  | `'train'` | Which dataset split to use (`train`, `validation`, or `test`) |

### Interactive PC to CAD Pipeline and Loss Visualization
In the notebook `notebooks/pc2cad.ipynb` one can interactively encode point clouds into latent representations, decode them into CAD-sequences and analyse, how the Command- and Argument-Loss are calculated in thorough detail. Furthermore all CAD-sequences can be exported to `.step` and `.stl` files and all temporary and final results are saved for later inspection.

## Evaluation
In order to evaluate the DeepCAD model in auto-encoding CAD-sequences and the PointNet++ and DeepCAD pipeline in reconstructing point clouds to CAD-models, there are two scripts provided by the authors of DeepCAD and modified for this project. Before calculating the metrics $ACC_{cmd}$ and $ACC_{args}$ first infer all samples (or whole sets) that you want to evaluate and place them within "path/to/output/dir". If you run `pc2cad.py` the outputs will be automatically saved in the model directory. 

Test the PointNet++ to get the MSE:
```bash
$ python code/test.py --model_path "path/to/model/from/root"
```
Infer all test point clouds to obtain the vectorized CAD-sequences. The output directory specified by `exp_name` is saved in the model directory under "results". Also it is the one to provide for the evaluation scripts:
```bash
$ python pc2cad.py --exp_name "recreate_metrics" --model_path "path/to/model/from/root" --phase "test"
```
To obtain the command and argument accuracy run:
```bash
$ python models/DeepCAD/evaluation/evaluate_ae_acc.py --src "path/to/output/dir"
```
To obtain the Chamfer Distance and Invalid Ratio run:
```bash
$ python models/DeepCAD/evaluation/evaluate_ae_cd.py --src "path/to/output/dir"
```

## Tests
The tests check that the datasets are complete and that the data aligns, i.e. that each point cloud path is matched with the equivalent latent representation and CAD sequence. Run them from the repository root, after exporting `PYTHONPATH`:
```bash
$ pytest tests/
```

## Known issues and corrections
An audit carried out before the dataset release found one correctness bug in the generated data and several issues that prevented the committed code from regenerating it. The released data is corrected; the entries below record what was wrong, what the effect was, and what changed.

### Extrusion / subsequence mispairing (corrected in the released data)
`code/create_extr_data.py` grouped the points of a cloud with `np.unique(labels)`, which returns only the labels that are actually present, and then used the *enumeration index* `j` to look up the CAD subsequence: `sequences[str(j)]`. When an extrusion lost all of its points to the proximity filter in `code/create_data.py`, its label was missing from `labels`, `np.unique` returned a sequence with a gap, and every surviving extrusion after the gap was paired with its neighbour's subsequence.

For a model whose surviving labels are `[0, 1, 3, 4, 6]`, files `_2`, `_3`, `_4` were stored with the subsequences of extrusions 2, 3, 4 instead of 3, 4, 6.

- **Scope:** 1,137 of 362,225 samples, spread over 425 of 177,776 models (a further 162 models lost only their *last* extrusion, which is harmless because no index shifts).
- **What was affected:** only the `extrusion_id` and `sequence` datasets in `pc_extrusion_labels/`. The `.ply` clouds in `pc_extrusion/` were always partitioned correctly and were never touched.
- **Correction:** the affected label files were rewritten in place with `python check_extrusion_pairing.py --data-root data --fix`. `code/create_extr_data.py` now returns the label values from `split_pc_by_labels` and uses them for both the lookup and the stored `extrusion_id`, so a regeneration produces the corrected pairing directly. Run `verify_extrusion_pairing.py` to confirm.

### The committed generator could not reproduce the released data
Three separate defects meant that running the published scripts as committed did not regenerate the published data. All three are fixed.

- **`N_POINTS` was 2048, but `pc_from_vec` holds 10,000 points.** `create_extr_data.py` asserts on the cloud size, so every sample raised — and because the exception was swallowed (below) the run reported no error and produced an empty output directory. `code/dataset.py` now separates `N_POINTS` (the generation size, 10,000) from `N_POINTS_TRAIN` (the training sample size, 2048, unchanged), the dataset classes take an `n_points` override, and the generation scripts state the size they need explicitly instead of inheriting a shared global.
- **Exceptions were swallowed.** Both generators wrapped their per-sample work in a bare `except Exception` that recorded the error into a dict and never failed the run, which is how the systematic failure above went unnoticed. The broad catch is kept — individual samples do legitimately fail on malformed CAD — but the scripts now print a running failure count, summarise `N of M samples failed` with the most common exception types, and exit non-zero above a 1% failure rate.
- **Data paths were hardcoded to `../data`** and both scripts had to be run from inside `code/`. Both now take `--data-root` (default `data`, relative to the current working directory, matching `pc2cad.py` and `code/train.py`) and are run from the repository root.

### Point subsampling was unseeded
The datasets subsample points with the global `random` module. During training this is a legitimate augmentation, but during generation it permuted the point order of the output files, so two runs of the generator produced clouds that were not byte-identical. The dataset classes now accept an optional `seed`; the default is still unseeded so training is unaffected, and the generation scripts pass a fixed seed.

## Other
### Experimental notebooks
The `notebooks` directory holds the exploratory notebooks used to develop and inspect the
scripts and models. They are kept for transparency and are not part of the reproduction
path described above; paths inside them are relative to the repository root.

### Report
The `report` directory contains the slides for the literature review that preceded this work.

## Citation
If you use this code or the accompanying dataset, please cite:

> Harb, S., Maboudi, M., and Gerke, M.: Extrusion Segmentation Strategy to improve CAD
> Reconstruction from Point Cloud, *The International Archives of the Photogrammetry,
> Remote Sensing and Spatial Information Sciences*, XLIX-B2-2026, 189-197, 2026.
> https://doi.org/10.5194/isprs-archives-XLIX-B2-2026-189-2026

```bibtex
@Article{isprs-archives-XLIX-B2-2026-189-2026,
  AUTHOR  = {Harb, S. and Maboudi, M. and Gerke, M.},
  TITLE   = {Extrusion Segmentation Strategy to improve CAD Reconstruction from Point Cloud},
  JOURNAL = {The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  VOLUME  = {XLIX-B2-2026},
  YEAR    = {2026},
  PAGES   = {189--197},
  URL     = {https://isprs-archives.copernicus.org/articles/XLIX-B2-2026/189/2026/},
  DOI     = {10.5194/isprs-archives-XLIX-B2-2026-189-2026}
}
```

## License
This repository is released under the MIT License; see [LICENSE](LICENSE).

### Third-party code
Two upstream projects are vendored under `models/` and remain under their own licenses:

- [DeepCAD](https://github.com/ChrisWu1997/DeepCAD) (Wu et al., ICCV 2021) — MIT,
  Copyright (c) 2022 Rundi Wu. Modifications made for this project are recorded in
  [models/DeepCAD/CHANGELOG.md](models/DeepCAD/CHANGELOG.md).
- [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) — MIT,
  Copyright (c) 2019 benny. Modifications are recorded in
  [models/Pointnet_Pointnet2_pytorch/CHANGELOG.md](models/Pointnet_Pointnet2_pytorch/CHANGELOG.md).

