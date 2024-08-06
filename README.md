This repository contains the code and visualization demo for our paper **"Diff-ExpertNet: A Deep Expert Framework Using Conditional Diffusion Models for Vessel Trajectory Generation"** submitted to [KDD] 2025.


## Repository Structure

- `code/`: Contains the source code for this paper.
  - `data_preprocessing/`: Scripts for AIS data preprocessing.
  - `models/`: Implementation models of our proposed framework.
  - `utils/`: Utility scripts.
- `map/`: Map image, ports and coastlines.
- `demo/`: Visualization demo about experimental results.



### Dataset

1. Download public AIS dataset from [source](https://web.ais.dk/aisdata/).
2. Details about AIS data preprocessing can be found in `code/data_preprocessing` directory.


## Visualization demo about our proposed vessel trajectory generation methods

We merged the vessel trajectories from the test set for January 2023 into a single day, and then provided a comparison between the ground-truth and generated results.
