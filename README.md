# ecglearning
Detection of cardiac arrhythmia using deep learning techniques.

## Getting started
### Conda environment
You can install the dependencies in the `environment.yml` file by running
```sh
$ conda env create -f environment.yml
```
To start the virtual environment and verify that the packages were installed correctly, run
```sh
$ conda activate ecgenv
$ conda list
```
The [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) has more information about manging environments. If you'd like to add packages to the environment, don't forget to update the environment.yml file with
```sh
conda env update -f environment.yml
```

### Tensorflow
To test that the TensorFlow 2.0 beta is up and running, start the `conda` environment (if you haven't already) and run `python`:
```sh
$ conda activate ecgenv
$ python
```
To check your tensorflow version:
```python
>>> import tensorflow as tf
>>> print(tf.__version__)
```
If you have an appropriate compute-capable NVIDIA gpu, you can check that it's working:
```python
>>> print("GPU Available:", tf.test.is_gpu_available())