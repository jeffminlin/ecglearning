# ecglearning
Detection of cardiac arrhythmia using deep learning techniques. This was a group project with the Graduate Data Science Organization at U.C. Berkeley with Jeffmin Lin, Alex Robson, Yao Tang, and Yang Ha under the guidance of Mike Osorio. See the [GDSO website](https://gdso.berkeley.edu/) for more information about the organization, the Data Science workshop, or to see the slides from our presentation.

## How to play with the data
### Data
You can find the data at [Kaggle](https://www.kaggle.com/shayanfazeli/heartbeat).

### Existing model architectures
You can examine the included notebooks in the `code` folder to see the architectures that we've played with. There is a very basic API which wraps Tensorflow Keras and some plotting utilities for the specific models in this project in the included `tools` package.

### Conda environment
If you're running Windows, you can install the dependencies in the `environment-win.yml` file by running
```sh
$ conda env create -f environment-win.yml
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
$ conda activate ecgenv
(ecgenv) $ conda install pandas
(ecgenv) $ conda install scikit-learn
(ecgenv) $ conda install tensorflow
(ecgenv) $ conda install tensorflow-gpu
(ecgenv) $ python -m pip install --upgrade tensorflow==2.0.0-beta1
(ecgenv) $ python -m pip install --upgrade tensorflow-gpu==2.0.0-beta1
(ecgenv) $ conda remove tensorflow --force
(ecgenv) $ conda remove tensorflow-gpu --force
```
Feel free to add/update your environment to the repo in a similar file, with
```sh
$ conda env export > environment-[OS].yml
```
where `[OS]` should be `mac`, `ubuntu`, etc. Make sure the saved encoding is `utf-8`.

To update your local environment with a new .yml file, run
```sh
$ conda env update -f environment-[OS].yml
```
where `[OS]` should be `mac`, `ubuntu`, etc.

The [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) has more information about manging environments. If you'd like to add packages to the environment file, update the `environment-[OS].yml` file with
```sh
$ conda env export > environment-[OS].yml
```
Make sure the encoding is `utf-8`, and push the result.

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