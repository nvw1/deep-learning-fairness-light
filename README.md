
## Impact of Data Imbalance on Fairness in Privacy Preserving Deep Learning
This final year project investigates the impact of model performance under small changes in the imbalance of minorities in the dataset and explores the resulting trends.
We aim to study why there are unexplained fluctuations in previously collected data and aim to help datascientist use measures of fairness in a reasonable way.

### Usage
Ensure to edit the wandb API key to your key (accessible from the wandb website) for recording results in the startup file and to alter the desired parameters in the `utils/params_celeba.yaml` file to create your custom experiment.


You can either use google colab(recommended) or run a local jupyter notebook by copying or running the contents of `startup.ipynb`

We use Python 3.7 and GPU Nvidia V100. <br />

The startup file will set up the enviornment and download the code from the repository [repo] (https://github.com/nvw1/deep-learning-fairness-light) and will download and unzip the pre-processed Dataset from a google cloud services bucket into its parents directory.<br />

Once this is completed it will utilise the configuration stored in `utils/params_celeba.yaml` to run execute the program through the entry file `running.py`.
The results will be accessible in the console as well as in the connected wandb account dashboard with its respective Tensorboard graphs. <br />

The startup file uses pre processed data set imbalances currently: [
    1,
    .999,
    .99,
    .98,
    .96,
    .95,
    .94,
    .92,
    .9,
    .85,
    .8,
    .7,
    .5
]
If you want to edit this you need to download a raw version of the celeba dataset and preprocess it wit the pre-process-data.ipynb jupyter notebook to your liking.

NOTE: when running there may be a warning related to Leaking Caffe 2 threadpool this is due to an optimisation step and is not effecting the program.


### Datasets
1. CelebA Datsset (from [here] (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html))

This has been pre processed and stored in a google cloud bucket.
If you want to pre-process it yourself use the `pre-process-data.ipynb` file and adjust the location to where your data is stored.

### Code Sources
We use `compute_dp_sgd_privacy.py`  and rdp_accountant copied from public [repo](https://github.com/tensorflow/privacy).

DP-FedAvg implementation is taken from public [repo](https://github.com/ebagdasa/backdoor_federated_learning).

Implementation of fairness.py execution and part of other files is based on [repo] (https://github.com/FarrandTom/deep-learning-fairness) and papers following below.

Implementation of DPSGD is based on TF Privacy [repo](https://github.com/tensorflow/privacy) and papers:


### Papers
https://arxiv.org/pdf/2009.06389v3.pdf
https://arxiv.org/pdf/2009.06389.pdf
https://arxiv.org/pdf/1812.06210.pdf


=======



# Requirements:
Gmail account for google colab and wandb for full funcitonality.
Subscription to colab pro or pro + may be required to run larger experiments as colab may time out early if you leave the page or are inactive.

Otherwise a ML docker image with a CUDA compatible GPU , a minimum of 50GB disk storage and 16GB RAM.
Time taken: 7h with a NVIDIA V100 GPU 16GB (5120 CUDA cores) and 100 MBit/s internet connection.

# Keeping Colab alive:
Code to keep google colab alive: only needed with non pro versions
function ConnectButton(){
    console.log("Connect pushed"); 
    document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
}
setInterval(ConnectButton,60000);

# Helpfull traces
`trace1` follows the flow of the dataset from preprocessing till being loaded into the dataloaders for training. This allows to debug and understand any logical issues that may occur when pre proccessing data.
`trace3` follows performance optimisations related to PyTorch.