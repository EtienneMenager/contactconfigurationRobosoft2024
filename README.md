# contactconfigurationRobosoft2024

This file contains the code used to obtain the results of the article "Learning control strategy in soft robotics through a set of configuration spaces" presented in Robosoft conference in 2024.


# Quick start

## How to use

The project use [SofaGym](https://github.com/SofaDefrost/SofaGym) framework v22.12. Don't forget to add the new environments to sofagym. The proposed paths do not necessarily correspond to the paths in your installation.

## Folders

This folder is divided into 4 subfolders:

* sofagym: Gym environment for the CartStemContact and the BarManipulator examples.
* data: script to recover data for the Selector training.
* scenarios: learning script for different scenarios and different robots.
* tools: script to define the different agents and the learning algorithm.

# Citing
If you use the project in your work, please consider citing it with:

```bibtex
@article{menager2024contactconfiguration,
  title={Learning control strategy in soft robotics through a set of configuration spaces},
  author={M{\'e}nager, Etienne and Duriez, Christian},
  booktitle={2024 IEEE International Conference on Soft Robotics (RoboSoft)}, 
  year={2024}
}
```

