import os
import json
import warnings
from time import time, sleep

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet as p

from gym_line_follower.track import Track
from gym_line_follower.track_plane_builder import build_track_plane
from gym_line_follower.bullet_client import BulletClient
from gym_line_follower.line_follower_bot import LineFollowerBot
from gym_line_follower.randomizer_dict import RandomizerDict


def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)


class LineFollowerEnv(gym.Env):
    metadata = {"render.modes": ["human", "gui", "rgb_array", "pov"]}

    SUPPORTED_OBSV_TYPE = ["points_visible", "points_latch", "points_latch_bool", "camera"]

    def __init__(self, gui=True, nb_cam_pts=8, sub_steps=10, sim_time_step=1 / 250,
                 max_track_err=0.3, power_limit=0.4, max_time=60, config=None, randomize=True, obsv_type="points_latch",
                 track=None, track_render_params=None):
        """
        Create environment.
        :param gui: True to enable pybullet OpenGL GUI
        :param nb_cam_pts: number of line representation points
        :param sub_steps: number of pybullet simulation steps per one environment step
        :param sim_time_step: pybullet simulation time step
        :param max_track_err: maximum distance of bot from the line before signaling episode done
        :param power_limit: limit the robot motor power, should be in range (0, 1) where 1 is not limited
        :param max_time: maximum episode time in seconds. Episodes finishes when max time is greater
        :param config: config dict. If none, 'bot_config.json' is loaded
        :param randomize: when True, track is generated randomly at each episode start,
        :param obsv_type: type of line observation generated:
                            "points_visible" - returns flattened array shape (3 * nb_cam_pts,) - each line point has
                                3 parameters [x, y, visibility] where visibility is 1.0 if point is visible in camera
                                window and 0.0 if not.
                            "points_latch" - returns flattened array shape (2 * nb_cam_pts,) if at least 2 line points
                                are visible in camera window, otherwise returns previous observation
                            "points_latch_bool" - same as "latch" with one additional value to indicate if line is
                                visible or not (0 or 1) - shape (2 * nb_cam_pts + 1)
                            "camera" - return (240, 320, 3) camera image RGB array
        :param track: Optional track instance to use. If none track is generated randomly.
        :param track_render_params: Track render parameters dict.
        """

        self.local_dir = os.path.dirname(os.path.dirname(__file__))

        if config is None:
            config_path = os.path.join(self.local_dir, "bot_config.json")
            self.config = RandomizerDict(json.load(open(config_path, "r")))
        else:
            self.config = config

        self.gui = gui
        self.nb_cam_pts = nb_cam_pts
        self.sub_steps = sub_steps
        self.sim_time_step = sim_time_step
        self.max_track_err = max_track_err
        self.speed_limit = power_limit
        self.max_time = max_time
        self.max_steps = max_time / (sim_time_step * sub_steps)

        self.randomize = randomize
        self.obsv_type = obsv_type.lower()
        self.track_render_params = track_render_params
        self.preset_track = track

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        if self.obsv_type not in self.SUPPORTED_OBSV_TYPE:
            raise ValueError("Observation type '{}' not supported.".format(self.obsv_type))

        if self.obsv_type == "points_visible":
            self.observation_space = spaces.Box(low=np.array([0.0, -0.2, 0.] * self.nb_cam_pts),
                                                high=np.array([0.3, 0.2, 1.] * self.nb_cam_pts),
                                                dtype=np.float32)
        elif self.obsv_type == "points_latch":
            self.observation_space = spaces.Box(low=np.array([0.0, -0.2] * self.nb_cam_pts),
                                                high=np.array([0.3, 0.2] * self.nb_cam_pts),
                                                dtype=np.float32)
        elif self.obsv_type == "points_latch_bool":
            low = [0.0, -0.2] * self.nb_cam_pts
            low.append(0.)
            high = [0.3, 0.2] * self.nb_cam_pts
            high.append(1.0)
            self.observation_space = spaces.Box(low=np.array(low),
                                                high=np.array(high),
                                                dtype=np.float32)
        elif self.obsv_type == "camera":
            self.observation_space = spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8)

        self.pb_client: p = BulletClient(connection_mode=p.GUI if self.gui else p.DIRECT)
        self.pb_client.setPhysicsEngineParameter(enableFileCaching=0)
        p.resetDebugVisualizerCamera(cameraDistance=2.6, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self.np_random = None
        self.step_counter = 0
        self.observation = []

        self._render_time = 0.

        self.follower_bot: LineFollowerBot = None
        self.track: Track = None

        self.position_on_track = 0.
        self.prev_track_distance = 0.
        self.prev_track_angle = 0.

        self.done = False

        self.plot = None
        self.seed()

    def reset(self):
        self.step_counter = 0
        self.config.randomize()

        self.pb_client.resetSimulation()
        self.pb_client.setTimeStep(self.sim_time_step)
        self.pb_client.setGravity(0, 0, -9.81)

        if self.randomize:
            if self.track_render_params:
                self.track_render_params.randomize()

        if self.preset_track:
            self.track = self.preset_track
        else:
            self.track = Track.generate(1.75, hw_ratio=0.7, seed=None if self.randomize else 4125,
                                        spikeyness=0.3, nb_checkpoints=500, render_params=self.track_render_params)

        start_yaw = self.track.start_angle
        if self.randomize:
            start_yaw += np.random.uniform(-0.2, 0.2)

        build_track_plane(self.track, width=3, height=2.5, ppm=1500, path=self.local_dir)
        self.pb_client.loadURDF(os.path.join(self.local_dir, "track_plane.urdf"))
        self.follower_bot = LineFollowerBot(self.pb_client, self.nb_cam_pts, self.track.start_xy, start_yaw,
                                            self.config, obsv_type=self.obsv_type)

        self.position_on_track = 0.

        if self.plot:
            plt.close(self.plot["fig"])
            self.plot = None

        self.done = False

        obsv = self.follower_bot.step(self.track)
        if len(obsv) < 1:
            return self.reset()  # TODO: maybe add recursion limit
        else:
            obsv = self.follower_bot.step(self.track)
            if self.obsv_type == "latch_bool":
                obsv = [obsv, 1.]
            return obsv

    def step(self, action):
        action = self.speed_limit * np.array(action)

        if self.done:
            warnings.warn("Calling step() on done environment.")

        reward = 0.

        for _ in range(self.sub_steps):
            self.follower_bot.apply_action(action)
            self.pb_client.stepSimulation()

        # Bot position updated here so it must be first!
        observation = self.follower_bot.step(self.track)

        if self.obsv_type == "points_visible":
            self.observation = observation

        elif self.obsv_type == "points_latch":
            if len(observation) == 0:
                observation = self.observation
            else:
                self.observation = observation

        elif self.obsv_type == "points_latch_bool":
            if len(observation) == 0:
                observation = [self.observation, 0.]
            else:
                self.observation = observation
                observation = [observation, 1.]

        elif self.obsv_type == "camera":
            self.observation = observation

        # Track distance error
        track_err = self.track.distance_from_point(self.follower_bot.pos[0])
        track_err_norm = track_err * (1.0 / self.max_track_err)

        self.position_on_track += self.track.length_along_track(self.follower_bot.prev_pos[0], self.follower_bot.pos[0])

        # Track progress
        checkpoint_reward = 1000. / self.track.nb_checkpoints
        if self.position_on_track - self.track.progress < 0.4:
            checkpoints_reached = self.track.update_progress(self.position_on_track)
            reward += checkpoints_reached * checkpoint_reward * (1.0 - track_err_norm) ** 2

        # Time penalty
        reward -= 0.2

        done = False
        if self.track.done:
            done = True
            print("TRACK DONE")
        elif abs(self.position_on_track - self.track.progress) > 0.5:
            reward = -100
            done = True
            print("PROGRESS DISTANCE LIMIT")
        elif track_err > self.max_track_err:
            reward = -100.
            done = True
            print("TRACK DISTANCE LIMIT")
        elif self.step_counter > self.max_steps:
            done = True
            print("TIME LIMIT")

        info = self._get_info()
        self.step_counter += 1
        self.done = done
        return observation, reward, done, info

    def render(self, mode='human'):
        if self.plot is None and mode in ["human", "rgb_array"]:
            global plt
            import matplotlib
            if mode == "rgb_array":
                matplotlib.use("agg")
                import matplotlib.pyplot as plt
            elif mode == "human":
                import matplotlib.pyplot as plt
                plt.ion()

            fig, (track_ax, win_ax) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
            track_ax.axis("off")
            # win_ax.axis("off")
            track_ax.set_aspect("equal")
            win_ax.set_aspect("equal")
            track_ax.set_xlim(-1.5, 1.5)
            track_ax.set_ylim(-1, 1)

            track_ax.plot(self.track.x, self.track.y, "k--")
            win_ax.plot(*self.follower_bot.cam_window.get_local_window().plottable, "m-")

            pos_line, = track_ax.plot(0, 0, "ro")
            win_line, = track_ax.plot(*self.follower_bot.cam_window.plottable, "m--")
            track_ref_line, = track_ax.plot(*self.follower_bot.track_ref_point.plottable, "c.")
            vis_pts_line, = win_ax.plot([], [], "c.")
            progress_line, = track_ax.plot([], [], "g--")

            self.plot = {"fig": fig,
                         "track_ax": track_ax,
                         "win_ax": win_ax,
                         "pos_line": pos_line,
                         "win_line": win_line,
                         "track_ref_line": track_ref_line,
                         "vis_pts_line": vis_pts_line,
                         "progress_line": progress_line}

        if mode in ["human", "rgb_array"]:
            # Plot data
            (x, y), yaw = self.follower_bot.get_position()
            self.plot["pos_line"].set_xdata(x)
            self.plot["pos_line"].set_ydata(y)

            self.plot["win_line"].set_xdata(self.follower_bot.cam_window.plottable[0])
            self.plot["win_line"].set_ydata(self.follower_bot.cam_window.plottable[1])

            if self.obsv_type == "visible":
                vis_pts = np.array(self.observation).reshape((-1, 3))
            elif self.obsv_type in ["latch", "latch_bool"]:
                vis_pts = np.array(self.observation).reshape((-1, 2))
            else:
                vis_pts = np.array(self.observation).reshape((-1, 2))

            self.plot["vis_pts_line"].set_xdata(vis_pts[:, 0])
            self.plot["vis_pts_line"].set_ydata(vis_pts[:, 1])

            self.plot["track_ref_line"].set_xdata(self.follower_bot.track_ref_point.plottable[0])
            self.plot["track_ref_line"].set_ydata(self.follower_bot.track_ref_point.plottable[1])

            self.plot["progress_line"].set_xdata(self.track.pts[0:self.track.progress_idx, 0])
            self.plot["progress_line"].set_ydata(self.track.pts[0:self.track.progress_idx, 1])

        if mode == "human":
            plt.draw()
        elif mode == "rgb_array":
            img = fig2rgb_array(self.plot["fig"])
            return img
        elif mode == "gui":  # Sleep to make GUI realtime
            while time() - self._render_time < 1 / 25:
                sleep(0.001)
            self._render_time = time()
        elif mode == "pov":
            return self.follower_bot.get_pov_image()
        else:
            super(LineFollowerEnv, self).render(mode=mode)

    def close(self):
        if self.plot:
            plt.close(self.plot["fig"])
            plt.ioff()
        return

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_info(self):
        (x, y), yaw = self.follower_bot.pos
        return {"x": x,
                "y": y,
                "yaw": yaw}

    def _get_velocity_along_track(self):
        (vx, vy), wz = self.follower_bot.get_velocity()
        v = np.array([vx, vy])
        x, y = self.follower_bot.track_ref_point.position
        track_vect = self.track.nearest_vector((x, y))
        # Project velocity vector to unit track vector
        return np.dot(v, track_vect)


class LineFollowerCameraEnv(LineFollowerEnv):

    def __init__(self):
        super(LineFollowerCameraEnv, self).__init__(obsv_type="camera")


if __name__ == '__main__':

    env = LineFollowerEnv(gui=True, nb_cam_pts=8, max_track_err=0.4, power_limit=0.4, max_time=600, obsv_type="points_latch")
    env.reset()
    for _ in range(100):
        for i in range(1000):
            obsv, rew, done, info = env.step((0., 0.))
            sleep(0.05)
            if done:
                break
        env.reset()
    env.close()
