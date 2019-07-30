# ecglearning
Detection of cardiac arrhythmia using deep learning techniques.

## Getting started
### Conda environment
If you're running Windows, you can install the dependencies in the `environment.yml` file by running
```sh
$ conda env create -f environment.yml
```
The installation may take a while as the TensorFlow beta installation using pip can be slow, so please be patient if it hangs for a bit after "Executing transaction: done".

To start the virtual environment and verify that the packages were installed correctly, run
```sh
$ conda activate ecgenv
$ conda list
```

If the above does not work, or you're on MacOS or Linux, you can install a similar environment with
```sh
$ conda create -n ecgenv python=3.7
$ conda install pandas
$ conda install scikit-learn
$ conda install tensorflow
$ conda install tensorflow-gpu
$ conda activate ecgenv
(ecgenv) $ python -m pip install upgrade tensorflow==2.0.0-beta1
(ecgenv) $ python -m pip install upgrade tensorflow-gpu==2.0.0-beta1
```

The [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) has more information about manging environments. If you'd like to add packages to the environment, don't forget to update the environment.yml file with
```sh
conda env update -f environment.yml
```


### TensorFlow
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