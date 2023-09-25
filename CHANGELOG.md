# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0](https://github.com/fabian57fabian/prototypical-networks-few-shot-learning) - 2023-09-25

### Added

- Stanford cars dataset with training fast usage
- Model loading in meta-train
- Changelog

### Fixed

- .

### Changed

- Readme description

### Removed

- .

# [1.0.0](https://github.com/fabian57fabian/prototypical-networks-few-shot-learning/releases/tag/v1.0.0) (2023-09-24)

### Added

- learn centroids script in src.core
- predict scirpt in src.core
- custom dataset option
- image channels option
- Continous integration tests on Github Actions
- added more tests on datasets and net
- CI tests on python 3.7-3.10
- Test badge in README
- Coveralls.io coverage bagde in README
- centroids, core
- version in src.__init__

### Fixed

- README pip install '-r'
- image channels not static
- urls for tests with new light test releases

### Changed

- train script secondary validation argument
- tran -> meta-train
- test -> meta-test
- moved some functions from src.core to src.utils
- how dataset is downlaoded to allow light tests
- workflow name to tests

### Removed

- .

# [0.2.0](https://github.com/fabian57fabian/prototypical-networks-few-shot-learning/releases/tag/v0.2.0) (2023-09-13)

### Added

- omniglot dataset
- AbstractClassificationDataset class
- yaml configuration save
- flowers102 dataset
- scripts to launch training
- results in README
- dataset description with images in README
- ùcosine distance
- training images
- added eval_each hyperparameter
- image_size param
- basic unit tests
- test function in src.core
- train_all bash script
- installation wiki in README
- presentation
- results graphs

### Fixed

- torch.nograd() in validation
- loss computation
- image size changed between mini_imagenet and omniglot
- distance computation bug
- flowers102 basic training size
- requirements troch and torchvision

### Changed

- mini imagenet dataset to extend AbstractClassificationDataset
- training script moved in src.core

### Removed

- download_imagenet bash script

# [0.1.0](https://github.com/fabian57fabian/prototypical-networks-few-shot-learning/releases/tag/v0.1.0) (2023-09-11)

### Added

- Basic README.md, .gitignore for Pycharm projects
- Prototypical networks Paper
- requirements.txt file for installation
- Prototypical network and loss
- DataLoader for meta-dataset
- mini_imagenet dataset in pre-release
- hyperparameters control on epochs, learning rate, NC, NQ, NS, episodes
- Validation every epoch
- model saving each X steps
- Tensorboard training summary

### Fixed

- .

### Changed

- .

### Removed

- .