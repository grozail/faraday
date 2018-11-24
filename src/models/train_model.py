import torch
from src.models import *
from src.data.gym.trajectory import ServoTrajectoryGenerator, RandomTrajectoryGenerationProcess, create_rd50
from src.data.gym.config import ControlConfiguration
from src.data.gym.environment import SingleServoEnv

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    process = RandomTrajectoryGenerationProcess(ServoTrajectoryGenerator(create_rd50(dt=0.0000625)), 6, 0.4)
    env = SingleServoEnv(process, ControlConfiguration(1, 20))
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], env.servo.max_current).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    ppo = PPO(model, optimizer, device, env, epochs=5, mb_size=50)
    ppo.train(100000, 200)