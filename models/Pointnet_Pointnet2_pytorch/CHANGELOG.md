# Changelog

The content of this directory was cloned from [yanx27 Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) repository on January 14th 2025. 
It is a pytorch implementation of [charlesq34 pointnet](https://github.com/charlesq34/pointnet) repository.
All notable changes to this project will be documented in this file.

## [0.1.5] - 2025-03-21
### Added
- Added new modified architecture ```get_model_tanh()``` in ```pointnet2_cls_msg.py```.
### Changed
- Changed ```get_model_new()``` to ```get_model_copy_author()``` in ```pointnet2_cls_msg.py```.

## [0.1.4] - 2025-03-18
### Added
- Added new modified architecture ```get_model_new()``` in ```pointnet2_cls_msg.py```.

## [0.1.3] - 2025-03-10
### Removed
- Removed final softmax layer from `pointnet2_cls_msg.py`
### Added
- Added MSE loss in `pointnet2_cls_msg.py` within the class `get_loss_mse`.
### Changed
- Changed name of `get_loss` to `get_loss_nll` for differentiation to `get_loss_mse` in `pointnet2_cls_msg.py`.

## [0.1.2] - 2025-01-16
### Removed
- Removed final softmax layer from `pointnet2_cls_ssg.py`

## [0.1.1] - 2025-01-15
### Added
- Added MSE loss in `pointnet2_cls_ssg.py` within the class `get_loss_mse`.
### Changed
- Changed name of `get_loss` to `get_loss_nll` for differentiation to `get_loss_mse` in `pointnet2_cls_ssg.py`.

## [0.1.0] - 2025-01-14
### Added
- Cloned Pointnet_Pointnet2_pytorch repository into `models/Pointnet_Pointnet2_pytorch`.
