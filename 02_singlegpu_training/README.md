# **Single-GPU Training**

Optimizing your script for a single GPU is a crucial first step before scaling to multiple GPUs. Inefficient single-GPU code may result in wasted resources and longer queue times when running multi-GPU jobs. This example demonstrates how to train a CNN on the MNIST dataset using a single GPU and profile the training process for performance improvements.

```bash
$ ssh <username>@<cluster_ip> -p 4422
$ mkdir <working_directory>
$ cd <working_directory>
$ git clone https://github.com/snsharma1311/SCA-2025-DistributedTraining.git
```
Replace <hpcaiuser> with user account created for participants
---

## **Step 1: Activate Conda Environment**

To simplify the setup, use a pre-configured Conda environment. Follow these steps:

### Setting Up Conda for Line Profiling
This step configures the environment to use `line_profiler` for analyzing the code and to download dataset.

```bash
$ module load miniconda
$ conda activate gujcost_workshop
```


## **Step 2: Download the MNIST Dataset**

Ensure the dataset is downloaded on the login node since compute nodes typically lack internet access:

```bash
(gujcost_workshop) $ cd DistributedTraining/02_singlegpu_training
(gujcost_workshop) $ python download_data.py
```



## **Step 3: Inspect and Profile the Script**

### Inspect the Training Script
Navigate to the directory and view the content of the training script:

```bash
(gujcost_workshop) $ cat mnist_model.py
```
Analyze the code for a better understanding of its structure and workflow.


### Script Sections

#### Importing Required Libraries
```python
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
```

#### Define the CNN Model
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

#### Initialize Model, Optimizer, and Scheduler
```python
# Initialize model, optimizer, and scheduler
model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
```

#### Configure DataLoader with CUDA Optimization
```python
if use_cuda:
    cuda_kwargs = {'num_workers': num_workers, 'pin_memory': True, 'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
```

#### Define Dataset Transformations
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize MNIST images
])
```

#### Initialize DataLoaders
```python
train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
```

#### Training Function with Profiling
```python
@profile
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ' 
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
```

#### Testing Function
```python
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')
```


## **Step 4: Slurm Script for Single-GPU Training**

Create a `slurm_submit.sh` file:

```bash
#!/bin/bash
#SBATCH --job-name=mnist_1GPU    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --partition=gpu          # Specify GPU partition for GPU Nodes
#SBATCH --reservation=hpcai      # Reservation incase of urgent requirement
##SBATCH --nodelist=rpgpu*        # Specify reservation GPU node name provided
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --output=logs_%j.out     # output logfile name
#SBATCH --error=logs_%j.err      # error logfile name
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

# which gpu node was used
echo "Running on host of RUDRA" $(hostname)

module purge #remove unneccesary loaded modules

#load the module
module load miniconda

#activate the environment
conda activate gujcost_workshop

#Try both the option, See the performance

kernprof -o ${SLURM_JOBID}_${SLURM_CPUS_PER_TASK}.lprof -l mnist_model.py --epochs=5
```

Submit the job:

```bash
(gujcost_workshop) $ sbatch slurm_submit.sh
```

### Monitor GPU and System Performance
Once allocated, monitor the GPU and system threads:
```bash
$ ssh <gpu_node>
$ watch -n 0.1 nvidia-smi
$ top -u <hpcusername>
```
This will help you observe GPU utilization and thread/process spawning during execution.



## **Step 5: Analyze Profiling Data**

After the job completes, analyze the profiling results:

```bash
(gujcost_workshop) $ python -m line_profiler -rmt <job_id>_<slurmtask>.lprof
```

---
---

## **Assignments**

Note: To edit the files use nano editor in linux environment

### use nano editor to edit files

```bash
(gujcost_workshop) $ nano mnist_model.py
```
and make changes as per assignment `ctrl + x` and type `yes` then `Enter`

1. Adjust `--cpus-per-task` to values like 2, 4, 6, 8, or 10 analyze the time and resource utilization (`slurm_submit.sh`).

2. Use the following line to test performance without GPU(`slurm_submit.sh`):
  ```bash
  kernprof -o ${SLURM_JOBID}_${SLURM_CPUS_PER_TASK}.lprof -l mnist_model.py --epochs=5 --no-cuda
  ```
3. Change the batchsize of dataloader to 128 and 256 see the performance(`slurm_submit.sh`):
  ```bash
  kernprof -o ${SLURM_JOBID}_${SLURM_CPUS_PER_TASK}.lprof -l mnist_model.py --epochs=5 --batch-size=128
  ```
5. Change `--ntasks=2` and set `--gres=gpu:2` see if code is runnin on multiple GPUs ?

---
---

## **How the Conda Environment Was Created**

```bash
$ module load miniconda
$ conda create --name workshop python=3.9 --yes
$ conda activate gujcost_workshop
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
$ conda install line_profiler --channel conda-forge
```


## **Summary**

Optimizing single-GPU training is a critical step before scaling to multiple GPUs. Efficient code ensures reduced resource usage and shorter queue times. By profiling and addressing bottlenecks, you can achieve higher GPU utilization and better performance.
[Go to Multi-GPU Data Parallel Training](../03_multigpu_dp_training/) for the next steps in distributed training.
