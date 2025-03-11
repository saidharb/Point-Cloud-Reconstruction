# Point Cloud to CAD Model Reconstruction using Deep Learning
GitHub repository for my Masters Thesis at IGP, TU Braunschweig

The goal of this project is to predict CAD sequences from input point clouds. 

## DeepCAD
The DeepCAD model from the [DeepCAD: A Deep Generative Network for Computer-Aided Design Models](https://arxiv.org/abs/2105.09492) ICCV 2021 by Wu et al. will serve in two purposes: First as an encoder for CAD sequences and second as the decoder for the point cloud latent representations. For the purpose of this project the original code will be modified (cf. [CHANGELOG.md](https://github.com/saidharb/Point-Cloud-Reconstruction/blob/main/models/DeepCAD/CHANGELOG.md)).

### Data
First you can download the data from the [DeepCAD dataset](http://www.cs.columbia.edu/cg/deepcad/data.tar) and extract it in the `data` folder in the main directory.

### Turn CAD Sequences into Point Clouds
To convert CAD sequences from the DeepCAD dataset into point clouds, use the `json2pc.py` script:
```bash
$ cd models/DeepCAD/dataset
$ python json2pc.py
```
This will create a new `pc_cad` directory within the `data` directory, containig the `.ply` point cloud files for the CAD sequences in the DeepCAD dataset.

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

## PointNet++
Until this point, the CAD-sequence dataset is downloaded, the CAD-sequences are converted to point clouds and the CAD-sequences are encoded into the latent space by the DeepCAD model. The next step is to train the PointNet++ model in a regression task on the point clouds with the latent representation of the CAD-sequences as targets. PointNet++ is from the paper [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413) NIPS 2017 by Qi et al. and for this project the [PyTorch version](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) will be used and modified (cf. [CHANGELOG.md](https://github.com/saidharb/Point-Cloud-Reconstruction/blob/main/models/Pointnet_Pointnet2_pytorch/CHANGELOG.md)). 

### Setup
As a good practice before executing the code, create a new conda environment and install all dependencies from the root repository:
```bash
$ conda create -n <name of environment>
$ conda activate <name of environment>
$ pip install -r requirements.txt
```
If there are problems with pip finding an appropriate version for open3d, use conda-forge to install it:
```bash
$ conda install -c conda-forge open3d
```

### Train PointNet++
In order to train PointNet++ a training script is developed, which will dynamically adapt to CPU or GPU. Before executing the training scrip two environment variables have to be set, in particular the `PYTHONPATH` and the `WANDB_API_KEY`, if training should be tracked using Weight and Biases. The API key can be obtained from your user account at WandB. Set the environment variables from the root repository like this:

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
| `--output_dir`     | `str`   | required     | Name of of the model save directory relative to root directory             |
| `--lr_type`     | `str`   | `step`, `step_adv` or `cosine`     | Learning rate type: 'step' for reducing learning rate on val_loss plateau or 'cosine' for cosine annealing with warm restarts|
| `--msg`     | `bool`   | False     | Use multi-scale-grouping instead of single-scale-grouping       |
                        

The training script will create a directory `models/trained_models` where all training runs are saved including the best and last model, checkpoint models, logging information and a csv file containing the metrics. The learning rate will be halfed, if the validation loss did not decrease for the number of epochs specified in `lr_patience`.

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

## PC to CAD Pipeline
The script ```pc2cad.py``` enables the full inference pipeline using PointNet++ as the encoder for point clouds and the DeepCAD decoder to decode them into CAD-sequences. Previously DeepCAD was trained on auto-encoding CAD-sequences and PointNet++ was trained on encoding point clouds into the same latent space as DeepCAD. The pre-trained DeepCAD model should be within ```data/latent/pretrained/model/ckpt_epoch1000.pth```. During inference metrics like the PointNet++ MSE-Loss and the DeepCAD Command- and Argument-Loss are recorded and predicted latent representation and CAD-sequence are saved within the model directory under ```results```. Run the script using the following command from root:

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
In the notebook ```notebooks/pc2cad.ipynb``` one can interactively encode point clouds into latent representations, decode them into CAD-sequences and analyse, how the Command- and Argument-Loss are calculated in thorough detail. Furthermore all CAD-sequences can be exported to ```.step``` and ```.stl``` files and all temporary and final results are saved for later inspection.

## Evaluation
In order to evaluate the DeepCAD model in auto-encoding CAD-sequences and the PointNet++ and DeepCAD pipeline in reconstruction point clouds to CAD-models, there are two scripts provided by the authors of DeepCAD and modfied for this project. Bevor calcualting the metrics $ACC_{cmd}$ and $ACC_{cmd}$ first infer all samples (or whole sets) that you want to evaluate and place them within "path/to/output/dir". If you run ```pc2cad.py``` the outputs will be automatically saved in the model directory. To obtain the command and argument accuracy run:
```bash
$ python models/DeepCAD/evaluation/evaluate_ae_acc.py --src "path/to/output/dir"
```
To obtain the Chamfer Distance and Invalid Ratio run:
```bash
$ python models/DeepCAD/evaluation/evaluate_ae_cd.py --src "path/to/output/dir"
```

## Pytest
In order to check if the datasets comprise of all data and the data alligns (i.e. the correct point cloud path is assigned to the equivalent latent representation and CAD-sequence). To run the tests (after exporting Pythonpath) enter the following in the command line from the root directory:
```bash
$ pytest tests/
```

## Other
### Experimental notebooks
In the code directory you can find experimental jupyter notebooks, which I use to explore and test out scripts and models.

### Report
In the report directory you can find the slides for my literature research. The whole repository is still work in progress.

