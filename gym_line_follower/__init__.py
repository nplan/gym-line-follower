from gym.envs.registration import register

register(
    id='LineFollower-v0',
    entry_point='gym_line_follower.envs:LineFollowerEnv',
    reward_threshold=700
)

register(
    id='LineFollowerCamera-v0',
    entry_point='gym_line_follower.envs:LineFollowerCameraEnv',
    reward_threshold=700
)