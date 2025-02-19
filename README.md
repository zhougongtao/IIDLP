## Introduction

This code supports the AAAI paper **Integrating Inference and Experimental Design for Contextual Behavioral Model Learning**

## Computing infrastructure requirements

This code has been tested on Windows using Python 3.
To train the designed network, we recommend using a CPU.  We used an Intel i7-12700 CPU with 32 GiB of memory.

## Environment Setup

- Python 3.x
- Required Python libraries are listed in the `requirements.txt` file.

## Experiment

To run the experiment, execute the `run.py` file. During the process, network weights will be saved in the `model_weights` folder, and the user decision dataset (`.pt` file) will be saved in the `dataset` folder. To ensure consistent results, we provide the initialization model weights.

The following functions in `run.py` offer the necessary commands:

```python
configs = manual_strategy_seed()
configs = I_ID_LP_RP_seed()
```

After generating the investor dataset, use the same network to output the results, which will be saved as `xxx.json` in the `result` folder:

```python
configs = NN_test()
```



