from gym.envs.registration import register

register(
    id='HomogeneousCoverage-v0',
    entry_point='envs.homogeneous_coverage:HomogeneousCoverageEnv',
    max_episode_steps=1000,
)

register(
    id='FlockingRelative-v0',
    entry_point='envs.flocking_relative:FlockingRelativeEnv',
    max_episode_steps=1000,
)
