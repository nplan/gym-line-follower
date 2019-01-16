import os

import numpy as np
import pybullet as p

from .reference_geometry import CameraWindow, ReferencePoint
from .line_interpolation import sort_points, interpolate_points
from .dc_motor import DCMotor
from .track import Track

JOINT_INDICES = {"left_wheel": 1,
                 "right_wheel": 2}
MOTOR_DIRECTIONS = {"left": -1,
                    "right": -1}


class LineFollowerBot:
    """
    Class simulating a line following bot with differential steering.
    """
    SUPPORTED_OBSV_TYPE = ["points_visible", "points_latch", "points_latch_bool", "camera"]

    def __init__(self, pb_client, nb_cam_points, start_xy, start_yaw, config, obsv_type="visible"):
        """
        Initialize bot.
        :param pb_client: pybullet client for simulation interfacing
        :param nb_cam_points: number of points describing line
        :param start_xy: starting x, y coordinates
        :param start_yaw: starting yaw
        :param config: configuration dictionary
        :param obsv_type: type of line observation generated:
                            "points_visible" - returns array shape (nb_cam_pts, 3)  - each line point has 3 parameters
                                [x, y, visibility] where visibility is 1.0 if point is visible in camera window
                                and 0.0 if not.
                            "points_latch" - returns array length nb_cam_points if at least 2 line points are visible in
                                camera window, returns empty array otherwise
                            "points_latch_bool" - same as "latch", se LineFollowerEnv implementation
                            "camera" - return (240, 320, 3) camera image RGB array
        """
        self.local_dir = os.path.dirname(__file__)
        self.config = config

        self.pb_client: p = pb_client
        self.bot = None

        self.prev_pos = ((0., 0.), 0.)
        self.pos = ((0., 0.), 0.)

        self.prev_vel = ((0., 0.), 0.)
        self.vel = ((0., 0.), 0.)

        self.cam_window: CameraWindow = None
        self.nb_cam_pts = nb_cam_points

        self.track_ref_point: ReferencePoint = None
        self.cam_target_point: ReferencePoint = None  # POV Camera target point
        self.cam_pos_point: ReferencePoint = None  # POC Camera position

        self.volts = 0.

        self.left_motor: DCMotor = None
        self.right_motor: DCMotor = None

        self.obsv_type = obsv_type.lower()
        if self.obsv_type not in self.SUPPORTED_OBSV_TYPE:
            raise ValueError("Observation type '{}' not supported.".format(self.obsv_type))

        self.reset(start_xy, start_yaw)

    def reset(self, xy, yaw):
        """
        Reload bot urdf, reposition bot, reinitialize camera window and other stuff.
        :param xy: starting xy coords
        :param yaw: starting yaw
        :return: None
        """
        self.bot = self.pb_client.loadURDF(os.path.join(self.local_dir, "follower_bot.urdf"), basePosition=[*xy, 0.0],
                                           baseOrientation=self.pb_client.getQuaternionFromEuler([0., 0., yaw]))
        self.pos = xy, yaw

        h = self.config["camera_window_height"]
        wt = self.config["camera_window_top_width"]
        wb = self.config["camera_window_bottom_width"]
        d = self.config["camera_window_distance"]
        win_points = [(d+h, wt/2), (d+h, -wt/2), (d, -wb/2), (d, wb/2)]

        self.cam_window = CameraWindow(win_points)
        self.cam_window.move(xy, yaw)

        tref_pt_x = self.config["track_ref_point_x"]
        self.track_ref_point = ReferencePoint(xy_shift=(tref_pt_x, 0.))
        self.track_ref_point.move(xy, yaw)

        cam_target_pt_x = self.config["camera_target_point_x"]
        self.cam_target_point = ReferencePoint(xy_shift=(cam_target_pt_x, 0.))
        self.cam_target_point.move(xy, yaw)

        cam_pos_pt_x = self.config["camera_position_point_x"]
        self.cam_pos_point = ReferencePoint(xy_shift=(cam_pos_pt_x, 0.))
        self.cam_pos_point.move(xy, yaw)

        nom_volt = self.config["motor_nominal_voltage"]
        no_load_speed = self.config["motor_no_load_speed"]
        stall_torque = self.config["motor_stall_torque"]
        self.left_motor = DCMotor(nom_volt, no_load_speed, stall_torque)
        self.right_motor = DCMotor(nom_volt, no_load_speed, stall_torque)

        self.volts = self.config["volts"]

        # Disable joint motors prior to using torque control
        self.pb_client.setJointMotorControl2(bodyIndex=self.bot, jointIndex=JOINT_INDICES["left_wheel"],
                                             controlMode=self.pb_client.VELOCITY_CONTROL, force=0)
        self.pb_client.setJointMotorControl2(bodyIndex=self.bot, jointIndex=JOINT_INDICES["right_wheel"],
                                             controlMode=self.pb_client.VELOCITY_CONTROL, force=0)

    def get_position(self):
        position, orientation = self.pb_client.getBasePositionAndOrientation(self.bot)
        x, y, z = position
        orientation = self.pb_client.getEulerFromQuaternion(orientation)
        pitch, roll, yaw = orientation
        return (x, y), yaw

    def _update_position_velocity(self):
        new_xy, new_yaw = self.get_position()
        self.cam_window.move(new_xy, new_yaw)
        self.track_ref_point.move(new_xy, new_yaw)
        self.cam_target_point.move(new_xy, new_yaw)
        self.cam_pos_point.move(new_xy, new_yaw)
        self.prev_pos = self.pos
        self.prev_vel = self.vel
        self.pos = new_xy, new_yaw
        self.vel = self.get_velocity()

    def get_velocity(self):
        linear, angular = self.pb_client.getBaseVelocity(self.bot)
        vx, vy, vz = linear
        wx, wy, wz = angular
        return (vx, vy), wz

    def _get_wheel_velocity(self):
        l_pos, l_vel, l_react, l_torque = self.pb_client.getJointState(self.bot, JOINT_INDICES["left_wheel"])
        r_pos, r_vel, r_react, r_torque = self.pb_client.getJointState(self.bot, JOINT_INDICES["right_wheel"])
        return l_vel, r_vel

    def step(self, track: Track):
        """
        Should be called after simulation step.
        Update camera window and other reference geometry, generate observation.
        :param track: Track object
        :return: observation, according to self.obsv_type
        """
        self._update_position_velocity()
        visible_pts = self.cam_window.visible_points(track.mpt)

        if self.obsv_type == "points_visible":
            if len(visible_pts) > 0:
                pts = self.cam_window.convert_points_to_local(visible_pts)
                pts = sort_points(pts, origin=self.track_ref_point.get_xy())
                pts = interpolate_points(pts, segment_length=0.025)
            else:
                pts = np.zeros((0, 2))

            observation = np.zeros((self.nb_cam_pts, 3), dtype=np.float32)
            for i in range(self.nb_cam_pts):
                try:
                    x, y = pts[i]
                except IndexError:
                    x = np.random.uniform(0.0, 0.2)
                    y = np.random.uniform(-0.2, 0.2)
                    vis = 0.0
                else:
                    vis = 1.0
                observation[i] = [x, y, vis]
            observation = observation.flatten().tolist()
            return observation

        elif self.obsv_type in ["points_latch", "points_latch_bool"]:
            if len(visible_pts) > 0:
                visible_pts_local = self.cam_window.convert_points_to_local(visible_pts)
                visible_pts_local = sort_points(visible_pts_local)
                visible_pts_local = interpolate_points(visible_pts_local, self.nb_cam_pts)
                if len(visible_pts_local) > 1:
                    observation = visible_pts_local.flatten().tolist()
                    return observation
                else:
                    return []
            else:
                return []

        elif self.obsv_type == "camera":
            return self.get_pov_image()

    def _set_wheel_torque(self, l_torque, r_torque):
        """
        Apply torque to simulated wheels.
        :param l_torque: left wheel torque in Nm
        :param r_torque: right wheel torque in Nm
        """
        l_torque *= MOTOR_DIRECTIONS["left"]
        r_torque *= MOTOR_DIRECTIONS["right"]
        self.pb_client.setJointMotorControl2(self.bot,
                                             jointIndex=JOINT_INDICES["left_wheel"],
                                             controlMode=self.pb_client.TORQUE_CONTROL,
                                             force=l_torque)
        self.pb_client.setJointMotorControl2(self.bot,
                                             jointIndex=JOINT_INDICES["right_wheel"],
                                             controlMode=self.pb_client.TORQUE_CONTROL,
                                             force=r_torque)

    def _power_to_volts(self, l_pow, r_pow):
        """
        Convert power to volts
        :param l_pow:
        :param r_pow:
        :return:
        """
        l_pow = np.clip(l_pow, -1., 1.)
        r_pow = np.clip(r_pow, -1., 1.)
        return l_pow * self.volts, r_pow * self.volts

    def apply_action(self, action):
        """
        Apply torque to simulated wheels.
        :param action:
        :return: None
        """
        l_volts, r_volts = self._power_to_volts(*action)
        l_vel, r_vel = self._get_wheel_velocity()
        l_vel *= MOTOR_DIRECTIONS["left"]
        r_vel *= MOTOR_DIRECTIONS["right"]
        l_torque = self.left_motor.get_torque(l_volts, l_vel)
        r_torque = self.right_motor.get_torque(r_volts, r_vel)
        self._set_wheel_torque(l_torque, r_torque)

    def get_pov_image(self):
        """
        Render virtual camera image.
        :return: RGB Array shape (240, 320, 3)
        """
        cam_x, cam_y = self.cam_pos_point.get_xy()
        cam_z = 0.095
        target_x, target_y = self.cam_target_point.get_xy()
        vm = self.pb_client.computeViewMatrix(cameraEyePosition=[cam_x, cam_y, cam_z],
                                              cameraTargetPosition=[target_x, target_y, 0.0],
                                              cameraUpVector=[0.0, 0.0, 1.0])
        pm = self.pb_client.computeProjectionMatrixFOV(fov=49,
                                                       aspect=320 / 240,
                                                       nearVal=0.0001,
                                                       farVal=1)
        w, h, rgb, deth, seg = self.pb_client.getCameraImage(width=320,
                                                             height=240,
                                                             viewMatrix=vm,
                                                             projectionMatrix=pm,
                                                             renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.array(rgb)
        rgb = rgb[:, :, :3]
        return rgb