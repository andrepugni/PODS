# PODS: Potential Outcomes for Deferring Systems

We provide here the code associated to the paper A causal framework for evaluating deferring systems.

## Setup

To install the required packages from our `environment.yml` file, run the following command:

```
conda env create -f environment.yml
```

To activate the environment, run the following command:

```
conda activate pods
```

## Running the code

To run the code, you can use the following command:

```
python train.py -data all --defer_system all --seed 42
```

This will train the models for all the datasets and all the deferring systems. 
The results will be saved in the `resultsRAW` folder.

To obtain the final estimates included in our paper, you can run the following command:

```
python test.py
```

This will produce the final estimates for all the datasets and all the deferring systems.
The results will be saved in the `results` folder.


If you do not want to train the models from scratch, you can download them from [here](https://www.dropbox.com/scl/fo/6rxx0sy4dq1c86aqgw5sr/AIdZeeWh6yeSohg4oLjiJKY?rlkey=1in2itm23tx1nh4jaaht4hxwm&dl=0).