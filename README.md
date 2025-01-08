# Point-Cloud-Reconstruction
GitHub repository for my Masters Thesis

The goal of this project is to predict CAD sequences from input point clouds. 

## DeepCAD
The DeepCAD model from the [DeepCAD: A Deep Generative Network for Computer-Aided Design Models](https://arxiv.org/abs/2105.09492) ICCV 2021 will serve in two purposes: First as an encoder for CAD sequences and second as the decoder for the point cloud latent representations.

### Data
First you can download the data from the [DeepCAD dataset](http://www.cs.columbia.edu/cg/deepcad/data.tar) and extract it in the `data` folder in the main directory.

### Turn CAD Sequences into Point Clouds
In order to obtain the point cloud representation of the CAD sequences provided by the DeepCAD dataset use the script `json2pc.py` from the main directory:
```
python -m models.DeepCAD.dataset.json2pc --data_root "data"
```
This will create a new `pc_cad` directory within the `data` directory, containig the `.ply` point cloud files for the CAD sequences in the DeepCAD dataset.

## Other

### Experimental notebooks
In the code directory you can find two experimental jupyter notebooks, which I use to explore and test out the DeepCAD and PointNet model.

### Report
In the report directory you can find the slides for my literature research. The whole repository is still work in progress.

