from gym.envs.registration import register

register(
    id='line-follower-v0',
    entry_point='gym_line_follower.envs:LineFollowerEnv',
    trials=10,
    reward_threshold=800
)
