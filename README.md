ATEC_competition_demo_code_2025
===

[//]: # (Gym-Rescue: )
**This project integrates Unreal Engine with OpenAI Gym for a rescue task based on [UnrealCV](http://unrealcv.org/).**  

# Introduction

**This repository is a specialized branch of the gym-unrealcv project, tailored to simulate rescue task scenarios. It provides an interactive reinforcement learning (RL) environment where agents can be trained and tested in virtual rescue missions.**

# Installation

## Dependencies

- UnrealCV

- Gym

- CV2

- Matplotlib

- Numpy

- Docker(Optional)

- Nvidia-Docker(Optional)

 
We recommend you use [anaconda](https://www.continuum.io/downloads) to install and manage your Python environment.

```CV2``` is used for image processing, like extracting object masks and bounding boxes. ```Matplotlib``` is used for visualization.




It is easy to install, just activate your python environment and install dependency package
```
pip install -r requirements.txt
```


## Run the baseline code
```
./run_main.sh
```
## Task Demonstration
We provide a task execution example to help participants better understand the task. Participants can:  
    - Control the agent via keyboard to complete the rescue mission ```--keyboard```.  
    - Observe a randomly controlled agent navigating the environment.  
By adding the parameters ```--render``` and ```--record_video```, participants can visualize the agent’s first-person perspective and save the entire observation sequence as an MP4 file, gaining a clear understanding of the success and failure criteria.
### Run a keyboard controlled agent, visualize and save to output.mp4
    - Use `i`, `j`, `k`, `l` for agent movement  
    - `1` for pick  
    - `2` for drop  
    - `e` for open the door  
    - `space` for jump  
    - `ctrl` for crouch  

```
python example/rescue_demo.py --render --record_video  --keyboard
```

### Run a random agent, visualize it
```
python example/rescue_demo.py --render 
```




