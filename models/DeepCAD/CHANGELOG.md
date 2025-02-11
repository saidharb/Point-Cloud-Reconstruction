# Changelog

The content of this directory was cloned from [ChrisWu1997 DeepCAD](https://github.com/ChrisWu1997/DeepCAD) repository on January 7th 2025. 
All notable changes to this project will be documented in this file.

## [0.1.5] - 2025-02-11
### Changed
- Changed `configAE.py` hard coded command line argument ```"proj_dir"``` to keyword argument in the class ```ConfigAE```.

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
