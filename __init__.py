from gym.envs.registration import register

register(
    id='four_room-v1',
    entry_point='shared_dynamic_envs.envs:Fourrooms',
)

register(
    id='chain-v0',
    entry_point='shared_dynamic_envs.envs:Chain',
)

register(
    id='narrow_hallway-v0',
    entry_point='shared_dynamic_envs.envs:Narrow_Hallway',
)

register(
    id='Ushape_hallway-v0',
    entry_point='shared_dynamic_envs.envs:Ushape_Hallway',
)

register(
    id='wide_hallway-v0',
    entry_point='shared_dynamic_envs.envs:Wide_Hallway',
)

register(
    id='maze-v0',
    entry_point='shared_dynamic_envs.envs:Maze',
)

register(
    id='fork-v0',
    entry_point='shared_dynamic_envs.envs:Fork',
)

register(
    id='lqr-v0',
    entry_point='shared_dynamic_envs.envs:LqrEnv',
)

register(
    id='lsn-v0',
    entry_point='shared_dynamic_envs.envs:SparseNaviEnv',
)