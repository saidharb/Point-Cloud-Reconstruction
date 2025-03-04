# Changelog

The content of this directory was cloned from [ChrisWu1997 DeepCAD](https://github.com/ChrisWu1997/DeepCAD) repository on January 7th 2025. 
All notable changes to this project will be documented in this file.

## [0.1.7] - 2025-03-04
### Changed
- Enables optional argument parsing when instantiating the ```ConfigAE``` class
- Transformed ```sketch_plane.origin``` from numpy array to list before passing it to ```gp_Pnt``` in ```visualize.py```
- Changed imports in ```evaluate_ae_acc.py``` to match the repository structure
- Changed all ```np.int``` to ```np.int32``` in ```evaluate_ae_acc.py``` and ```test.py```
- Changed imports in ```evaluate_ae_cd.py``` to match the repository structure
- Changed the ```PC_ROOT``` in ```evaluate_ae_cd.py``` to match the repository structure


## [0.1.6] - 2025-02-14
### Changed
- Changed ```np.float``` to ```np.float64``` in ```export2step.py```
### Added
- Printing of exception in ```export2step.py```

## [0.1.5] - 2025-02-11
### Changed
- Changed `configAE.py` hard coded command line argument ```"proj_dir"``` to keyword argument in the class ```ConfigAE```.
- In the ```if extended:``` statment in ```_get_key_padding_mask``` a ```.clone()``` was added to the second tensor in order to mitigate the ```RuntimeError``` when using a batch size of one. Also there I have a doubt that the ```if extended:``` statement works correctly, as it actually does not include the final EOS command into the padding mask.

## [0.1.4] - 2025-02-05
### Changed
- Changed imports of modules to new repository structure
- Hard coded command line arguments
### Removed
- Removed tensorboard dependency
### Added
- Added abstract method and dependency on ABC in 'base.py`

## [0.1.3] - 2025-01-10
### Changed
- Removed corrupt index from split in `json2pc.py`: The index '0011/00116212' always caused a segmentation error when creating `.ply` files from `.json` files. Therefore it is removed from `all_data['train']`

## [0.1.2] - 2025-01-09
### Added
- `self.device` attribute to the `BaseTrainer` class which dynamically determines if cuda is available or cpu should be used.
### Changed
- Replaced all occurences of `.cuda()` with `.to(self.device)` in the `BaseTrainer` and `TrainerAE` class to dynamically use cuda or cpu.

## [0.1.1] - 2025-01-08
### Changed
- `DATA_ROOT` to new data root path

## [0.1.0] - 2025-01-07
### Added
- Cloned DeepCAD repository into `models/DeepCAD`.
