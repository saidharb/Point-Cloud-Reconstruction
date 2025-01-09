# Changelog

The content of this directory was cloned from [ChrisWu1997 DeepCAD](https://github.com/ChrisWu1997/DeepCAD) repository on January 7th 2025. 
All notable changes to this project will be documented in this file.

## [0.1.2] - 2025-01-09
### Added
- `self.device` attribute to the `BaseTrainer` class which dynamically determines if cuda is avaialble or cpu should be used.
### Changed
- Replaced all occurences of `.cuda()` with `.to(self.device)` in the `BaseTrainer` and `TrainerAE` class to dynamically use cuda or cpu.

## [0.1.1] - 2025-01-08
### Changed
- `DATA_ROOT` to new data root path

## [0.1.0] - 2025-01-07
### Added
- Cloned DeepCAD repository into `models/DeepCAD`.
