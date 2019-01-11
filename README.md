# Gym-Line-Follower

Line follower robot simulator for OpenAI Gym.

<img src="media/sim_env.gif" width="500">

## Introduction
Gym-Line-Follower is a simulator for a line following robot.
It is designed for developing line following algorithms using 
reinforcement learning. The simulator is written in Python and uses Pybullet engine for accurate
physics simulation. Gym-Line-Follower is fast and customizable. It currently supports differential
drive robots.

The line is represented by points inside a field-of-view window in front of the follower robot, as it would be
seen by a forward-facing camera on board. Rendering of a point-of-view camera image is supported.

<img src="media/line_representation.png" width="450">
Left: line track with progress marked green, Right: Camera field of view with line representation points in cyan.

---

This simulator was created as a part of masters thesis at 
[Faculty of Mechanical Engineering, University of Ljubljana](https://www.fs.uni-lj.si/).



## Installation
Requirements:
- Python 3.5+
- OpenAI Gym
- NumPy
- Shapely
- Pybullet
- OpenCV

```
git clone https://github.com/nplan/gym-line-follower.git
cd gym_line_follower
pip3 install gym_line_follower -e
```
> Requirements will be installed automatically. By using ````-e```` the environment is
installed in place and is editable.

## Usage
Load the environment as usual.
``` python
import gym
env = gym.make("LineFollower-v0")
```

 ## Environments
 ### LineFollower-v0
 Line follower is placed at the beginning of a closed loop line track. The objective is to follow the track as quickly
 and accurately as possible. Progress is measured in percent of track driven. Episode finishes when the track
 is completed (whole loop driven), the follower gets to far from the track or when the follower goes to far from
 the last progress point in the wrong direction.
 
 **Randomization:**
 Line track is generated randomly at the start of each episode
 and the starting angle of follower bot relative to the track is sampled from interval *(-0.2, 0.2) rad*.
 Other parameters are fixed and are specified in files ```bot_config.json``` and ```follower_bot.urdf```.
 
 **Observation Space:**
 *Box(8, 2)* - 8 points with *(x, y)* coordinates representing line. Origin of the coordinate system is the robots
 center of rotation. *x* in range *(0, 0.3)*, *y* in range *(-0.2, 0.2)*.
 
 **Action Space:**
 *Box(2)* - 2 values representing left and right motor power in range *(-1, 1)*.
 
 **Reward:**
 Track is split in 500 checkpoints that must be reached for the episode to complete successfully.
 When a checkpoint is reached, a reward is calculated using the following equations:
 ```
 checkpoint_reward = 1000. / nb_checkpoints
 track_error_norm = track_err * (1.0 / max_track_error)
 reward = checkpoint_reward * (1.0 - track_error_norm) ** 2
 ```
 Where ```nb_checkpoints``` is number of track checkpoints (500), ```track_error``` is distance between follower bot 
 center of rotation and the closes point on track, ```max_track_error``` is maximum allowed *track_error*.
 
 At each step a value of 0.2 is subtracted from reward to encourage quick completion of the track.
 
 If the episode finishes before the track is complete, a reward of -100 is given.
 
 
 ## Customized environments
 Only one environment is currently registered. Custom environments can be quickly built by making the environment
 using the class constructor with arguments.
 
 ``` python
 from gym_line_follower import LineFollowerEnv
 env = LineFollowerEnv(gui=False, nb_cam_pts=8, max_track_err=0.3, speed_limit=0.2,
                       max_time=100, randomize=True, obsv_type="latch")
 ```
 > Description of arguments is provided in source code.
 
 ## Configuration and Randomization

 The simulator can be configured with parameters inside the file ```bot_config.json```. Randomization at the beginning of
 each episode can be enabled for each parameter. You must format the json file according to the following rules:
 
 >If value randomization at a key is desired, provided value must be a dict with entries:
 > - 'range': [a, b] - random uniform sample in range a, b
 > - 'choice': [a, b, c, d, ...] - random choice of one value from the list
 > - 'default': a - default value
 
 >   Default value must always be provided. One of keys 'range' or 'choice' must be provided.
 >   If value at key is not a dict no randomization is performed.
 
 Randomization is enabled with the ```randomize``` parameter.
 
 >*WARNING*: Randomization is not yet working to the full extent. Some parameters are still hardcoded
 inside the file ```follower_bot.urdf```. The code to dynamically build this file from parameters inside 
 ```bot_config.json``` will be added in the future.
 

 ## Render options
 - ```"human"``` - display a live *matplotlib* plot with 2D representation of line track and representation points.
 <img src="media/human_render_mode.png" width="400">
 
 - ```"rgb_array"``` - same as *"human"* but return an RGB image array instead of displaying plot.  
 
 - ```"pov"``` - return an RGB image array from a forward facing point-of-view camera.
 <img src="media/pov_render_mode.png" width="300">

 - ```"gui"``` - make the *pybullet* GUI run at real-time FPS.
 
 ## Track rendering randomization
 Line track can be rendered with different colors and background (floor) textures. This is not enabled by
 default. Desired parameters can be set in ```track_render_config.json``` file, which must be formatted in the same 
 way as described in *Configuration and Randomization* section. 
 
 <img src="media/track_render_options.gif" width="400">
 
 Track rendering randomization
 can be enabled as following:
 ``` python
 render_params = RandomizerDict(json.load(open("track_render_config.json", "r")))
 env = LineFollowerEnv(track_render_params=render_params)
 ```
 
 >This future can be used in combination with ```"pov"``` render mode for automatic generation of labeled image data, 
 that can be used for training a line detector neural network.