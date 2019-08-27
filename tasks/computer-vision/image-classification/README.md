# Image Classification
We trained and evaluated a simple CNN for MNIST dataset using tensorflow and mlflow.

## Setup environment and install packages

* Setup python virtual environment:
```bash
virtualenv -p python venv
```

* Install requirements:
```bash
pip install -r requirements.txt
```

* Install tensorflow 1.14 CPU version for python 2:
```bash
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.14.0-cp27-none-linux_x86_64.whl
```

## How to run
After installing all dependencies you are ready to train and evaluate the model. Move inside tasks/computer-vision
folder and run:
```
mlflow run image-classification
```

You can also enable early stopping like that:
```
mlflow run image-classification -P early_stopping=True
```

Running again the script with the same model_dir, you fine-tune on the previous model. So if you want to train
from scratch delete the previous model_dir or feed a different path from the previous one.

After training we can view using Tensorboard and Mlflow:
```bash
tensorboard --logdir {model_dir}
mlflow ui
```