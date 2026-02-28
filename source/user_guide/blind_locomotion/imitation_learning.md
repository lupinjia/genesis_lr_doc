# Imitation Learning

## DeepMimic
We have implemented [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/index.html) for Unitree G1 robot. To use it, you can follow below steps

### 1. Prepare Retargetted Data
We have validated retargetted data from [GMR](https://github.com/YanjieZe/GMR), you can follow its instructions to generate retargetted reference motion for g1_29dof.

Then you should paste the retargetted data to `LeggedGymEx/resources/reference_motion/`.

### 2. Process Retargetted Data
To use the retargetted data in our framework, you should process it using `legged_gym/scripts/process_reference_motion.py`:

```python
python legged_gym/scripts/process_reference_motion.py --task=g1_mimic --motion_file=name_of_your_refenrece_motion.pkl

# for example
python process_reference_motion.py --task=g1_mimic --motion_file=02_01_walk_stageii_60hz.pkl
```

When processing, the program will visualize the motion in the simulator. By default, we use reference motion at 60Hz and the control frequency of the policy is also 60Hz.

The processed motion will be saved as a .pkl file under `LeggedGymEx/resources/reference_motion/`. The simulator name in its name indicate the simulator where it is generated.

### 3. Training the policy

Then you can start training by executing `python legged_gym/scripts/train.py --task=g1_mimic --headless --motion_file=name_of_your_processed_motion.pkl` (For example: `python legged_gym/scripts/train.py --task=g1_mimic --headless --motion_file=02_01_walk_stageii_60hz_isaacgym.pkl`).

:::{note}
Because link sequences in IsaacGym/Genesis/IsaacLab are different, please make sure you use the reference motion generated from the same simulator when training.
:::

After the training is over, you can see the result using `python legged_gym/scripts/play.py --task=g1_mimic --motion_file=name_of_your_processed_motion.pkl --load_run=loaded_training_session`. Below are some demos: 

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/lupinjia/genesis_lr_doc/raw/refs/heads/main/source/_static/videos/g1_mimic_walk_isaaclab.mp4" type="video/mp4">
</video>

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/lupinjia/genesis_lr_doc/raw/refs/heads/main/source/_static/videos/g1_mimic_run_isaacgym.mp4" type="video/mp4">
</video>

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/lupinjia/genesis_lr_doc/raw/refs/heads/main/source/_static/videos/g1_mimic_dance_isaacgym.mp4" type="video/mp4">
</video>