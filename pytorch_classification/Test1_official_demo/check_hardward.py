import torch
# check cuda version in pytorch (conda environment)
print(torch.version.cuda)

# check cuda version in y9000k: nvcc -V    or    nvcc --version

# check torch version
print(torch.__version__)

# check cudnn version
print(torch.backends.cudnn.version())

# check conda version: conda -V


