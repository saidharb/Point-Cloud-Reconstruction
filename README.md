# Point-Cloud-Reconstruction
GitHub repository for my Masters Thesis

The goal of this project is to predict CAD sequences from input point clouds. 

## DeepCAD
The DeepCAD model from the [DeepCAD: A Deep Generative Network for Computer-Aided Design Models](https://arxiv.org/abs/2105.09492) ICCV 2021 by Wu et al. will serve in two purposes: First as an encoder for CAD sequences and second as the decoder for the point cloud latent representations.

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

## Other
### Experimental notebooks
In the code directory you can find two experimental jupyter notebooks, which I use to explore and test out the DeepCAD and PointNet model.

### Report
In the report directory you can find the slides for my literature research. The whole repository is still work in progress.

