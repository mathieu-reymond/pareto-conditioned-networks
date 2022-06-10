# Pareto Conditioned Networks

This repository contains the code used for:

Reymond, M., Bargiacchi, E., & Nowé, A. (2022, May). Pareto Conditioned Networks. In Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems (pp. 1110-1118).

You can read the paper [here](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1110.pdf).

## Dependencies

The code requires `Python3.7+` as well as `torch` for the neural networks, `gym` for the environments, `h5py` for logging and `opencv-python` for preprocessing of image-observations.

## How to run

Here is how you run PCN on Deep Sea Treasure:

```
python main_pcn.py --env dst
```

This will create a log directory in `/tmp/pcn`. It also contains checkpoints of the learned policies.
You can then execute any of the learned policies as follows:

```
python eval_pcn.py <logdir>
```

Optionally, you can add an `--interactive` flag if you want to manually select policies.
