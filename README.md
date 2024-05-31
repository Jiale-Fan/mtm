# Masked Trajectory Models

## Environments and dependencies
*This is very hard to setup; Setting up on a different machine may differ from the following instructions.*

Ubuntu 22.04 x86_64 Python 3.10 
Anaconda as the environment manager.
NVIDIA GPU is required.

Softgym: https://github.com/Jiale-Fan/softgym , branch master
Softagent: https://github.com/Jiale-Fan/softagent, branch working
Please use the master branch of these modified versions of libraries instead of the original repos of the arthors.
Install them as packages using pip in the editable mode `pip install -e .`

To configure, first create a python environment using the `environment.yml` in this repo, and then install dependencies in the `requirements.txt` from the `SoftGym` forked repo. 
Build pyflex using docker environment following the notes https://danieltakeshi.github.io/2021/02/20/softgym/. The only difference is, instead of building using Python 3.6, use the Python 3.10 env we just created. Make sure the docker env has access to GPU resources.

## Usage

1. Create random initial configurations:
   At the root of package softgym, run
   ```python
  python examples/random_env.py --env_name RopeFlatten
   ```
2. Train CURL policy
  At the root of package softagent, run
   ```python
      python experiments/run_curl.py
   ```
  Collect the dataset: 
  specify the right path in script `softagent/curl/collect_data.py`, then run the script. 
3. Train MTM policy:
Choose either branch image or main (keypoints as observation)
  At the root of this package, mtm, run
  ```
python research/mtm/train_softgym.py
```
To evaluate the trained model in closed-loop way, run
  ```
python research/mtm/eval_softgym_closed.py

```
  
