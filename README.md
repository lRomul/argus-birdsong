# Cornell Birdcall Identification

Source code of solution for [Cornell Birdcall Identification](https://www.kaggle.com/c/birdsong-recognition) competition.

## Solution 

Key points: 
* Log-scaled mel-spectrograms
* CNN models
* Low public/private score :)

## Quick setup and start 

### Requirements 

*  Nvidia drivers, CUDA >= 10.2, cuDNN >= 7
*  [Docker](https://www.docker.com/), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 

The provided Dockerfile is supplied to build an image with CUDA support and cuDNN.


### Preparations 

* Clone the repo, build docker image. 
    ```bash
    git clone https://github.com/lRomul/argus-birdsong.git
    cd argus-birdsong
    make build
    ```

* Download and extract [dataset](https://www.kaggle.com/c/birdsong-recognition/data) to `data` folder.

### Run

* Run docker container 
```bash
make
```

* Create folds split 
```bash
python make_folds.py
```

* Train model
```bash
python train.py --experiment train_001
```
