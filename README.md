Our code for the paper: "WeakSupCon: Weakly Supervised Contrastive Learning for Encoder Pre-training".
The paper is accepted by MICCAI 2025 workshop on Efficient Medical AI.
The paper is available at https://arxiv.org/abs/2503.04165
The running environment in our experiments is Python 3.9.0, PyTorch 1.12.1+cu113, torchvision 0.13.1+cu113.
In our experiments, we only tested running on one GPU. (The GPU has 48 GB GPU memory). With the same batch size, it is recommended to run on one GPU with large GPU memory (rather than on multiple GPUs with small GPU memory for each single GPU), because the SimCLR loss will include more negative samples in each GPU. 
