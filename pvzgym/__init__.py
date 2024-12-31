from gymnasium.envs.registration import register

register(
    id='PlantsVsZombies',
    entry_point='pvzgym.envs:PvZEnv',
)
