# 🛠️Installation

## Prerequisites

Below table shows the recommended (tested) requirements of the commmmputer for running this framework.

| Component | Recommended (Tested) |
|-----------|-------------|
|    CPU    |Intel Core i9|
|    GPU    |   RTX 3080  |
|     OS    | Ubuntu 22.04|
|   Python  |     >=3.8   |
|Nvidia Driver|   >=535   |

Genesis_lr incoporates two simulators into one framework. One can choose either simulator to use, with each simulator requiring a seperate conda environment due to the limitation of python versions. Below is the recommended (tested) environment setting for two simulators:

| Component |  IsaacGym   |   Genesis   |
|-----------|-------------|-------------|
|  Python   |    3.8      |    >=3.10   |
|  Nvidia Driver |   535  |     535     |
|  PyTorch  | 2.4.1+cu121 | 2.7.1+cu118 |

## Direct Installation

### IsaacGym

```bash
# 1. Create a conda environment with python3.8
conda create -n lr_gym python=3.8
conda activate lr_gym
# 2. Install Pytorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# 3. Download IsaacGym Preview4 under /home/username
cd /home/username
wget https://developer.nvidia.com/isaac-gym-preview-4 \
    && tar -xf isaac-gym-preview-4 \
    && rm isaac-gym-preview-4
# Substitute np.float with np.float32 to resolve compatibility
find isaacgym/python -type f -name "*.py" -exec sed -i 's/np\.float/np.float32/g' {} +
# Install isaacgym in this environment
cd isaacgym/python && pip install -e . && cd ../..
# 4. Install genesis_lr with isaacgym
git clone https://github.com/lupinjia/genesis_lr.git
cd genesis_lr && pip install -e ".[isaacgym]"
# 5. Test the installation
python legged_gym/scripts/train.py --task=go2 --num_envs=100
```
If a window like below appears, the installation is successful.

```{figure} ../../_static/images/isaacgym_installation_success.png
```

### Genesis

```bash
# 1. Create a conda environment with python3.10
conda create -n lr_gen python=3.10
conda activate lr_gen
# 2. Install Pytorch
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
# 3. Install genesis_lr with genesis
git clone https://github.com/lupinjia/genesis_lr.git
cd genesis_lr && pip install -e ".[genesis]"
# 4. Test the installation
python legged_gym/scripts/train.py --task=go2 --num_envs=100
```
If a window like below appears, the installation is successful.

```{figure} ../../_static/images/genesis_installation_success.png
```

:::{note}
Comment following lines in `_set_dofs_info()` of `Genesis/genesis/engine/solvers/rigid/rigid_solvers_decomp.py`:

```python
elif name == "armature":
        kernel_set_dofs_armature(tensor_list[0], dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
        # qs_idx = torch.arange(self.n_qs, dtype=gs.tc_int, device=gs.device)
        # qpos_cur = self.get_qpos(envs_idx=envs_idx, qs_idx=qs_idx, unsafe=unsafe)
        # self._init_invweight_and_meaninertia(envs_idx=envs_idx, force_update=True, unsafe=unsafe)
        # self.set_qpos(qpos_cur, qs_idx=qs_idx, envs_idx=envs_idx, unsafe=unsafe)
```

If errors like this appears:
```bash
GenesisException: Links and dofs must be batched to selectively update invweight and meaninertia for some environment
```

:::



Finally, you need to register a wandb account and set the environment variable:
```bash
export WANDB_API_KEY=<your_api_key>
```

## Optional Installation

### Sim2Sim Support

Deploying the policy to another simulator can effectively test the robustness of the policy. Also, the code used for sim2sim can oftern be deployed directly to the real robot. To avoid potential collapse on the real robot, it's better to first test the deployment code in simulation. 

Since deployment codes are usually written in C++, a simulator which supports C++ interface is ideal. We offer a sim2sim framework in mujoco based on [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2), [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) and [LibTorch](https://pytorch.org/).

You can install this [go2_deploy](https://github.com/lupinjia/go2_deploy/tree/main) repo according to the instructions in README.md.