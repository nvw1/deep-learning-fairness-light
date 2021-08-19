
## Impact of Data Imbalance on Fairness in Privacy Preserving Deep Learning
This final year project investigates the impact of model performance under small changes in the imbalance of minorities in the dataset and explores the resulting trends. 

We aim to study why there are unexplained fluctuations in previously collected data and aim to help datascientist use measures of fairness in a reasonable way.




### Usage
On your colab project (or linux machine with ML image) run the `startup.ipynb` file<br />
We use Python 3.7 and GPU Nvidia V100. <br />

The startup file will set up the enviornment and download the code from the repository [repo] (https://github.com/nvw1/deep-learning-fairness-light) and will download and unzip the pre-processed Dataset from a google cloud services bucket into its parents directory.<br />

Ensure to now edit the wandb API key to your key (accessible from the wandb website) for recording results and to alter the desired parameters in the `utils/params_celeba.yaml` file to create your custom experiment.

Once this is completed it will utilise the configuration stored in `utils/params_celeba.yaml` to run execute the program through the entry file `running.py`.
The results will be accessible in the console as well as in the connected wandb account dashboard with its respective Tensorboard graphs. <br />



### Datasets
1. CelebA Datsset (from [here] (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html))

### Code Sources
We use `compute_dp_sgd_privacy.py` copied from public [repo](https://github.com/tensorflow/privacy).

DP-FedAvg implementation is taken from public [repo](https://github.com/ebagdasa/backdoor_federated_learning).

Implementation of playing.py execution is based on [repo] (https://github.com/FarrandTom/deep-learning-fairness) and papers following below.

Implementation of DPSGD is based on TF Privacy [repo](https://github.com/tensorflow/privacy) and papers:

### Paper
https://arxiv.org/pdf/2009.06389v3.pdf
https://arxiv.org/pdf/2009.06389.pdf



=======



# Requirements:
Gmail account for google colab and wandb for full funcitonality.
Subscription to colab pro or pro + may be required to run larger experiments as colab may time out early if you leave the page or are inactive.

Otherwise a ML docker image with a CUDA compatible GPU , a minimum of 50GB disk storage and 16GB RAM.
Time taken: 5h with a NVIDIA V100 GPU 16GB (5120 CUDA cores) and 100 MBit/s internet connection.
