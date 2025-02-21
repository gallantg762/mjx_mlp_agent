# base on mjx's sample rl_gym.py

from typing import Dict, List, Optional
import gym
import mjx.const
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical
import mjx
from mlp_agent import MLPAgent # agent with trained model
from tqdm import tqdm

# gym must be 0.25.0+ to use reset(return_info=True)
gym_version = [int(x) for x in gym.__version__.split(".")]
assert (
    gym_version[0] > 0 or gym_version[1] >= 25
), f"Gym version must be 0.25.0+ to use reset(infos=True): {gym.__version__}"

class PolicyNetwork(nn.Module):
    def __init__(self, obs_size=107*34, n_actions=181, hidden_size=107*34):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)

class GymEnv(gym.Env):
    def __init__(
        self, opponent_agents: List[mjx.Agent], reward_type: str, done_type: str, feature_type: str
    ) -> None:
        super().__init__()
        self.opponen_agents = {}
        assert len(opponent_agents) == 3
        for i in range(3):
            self.opponen_agents[f"player_{i+1}"] = opponent_agents[i]
        self.reward_type = reward_type
        self.done_type = done_type
        self.feature_type = feature_type

        self.target_player = "player_0"
        self.mjx_env = mjx.MjxEnv()
        self.curr_obs_dict: Dict[str, mjx.Observation] = self.mjx_env.reset()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = True,
        options: Optional[dict] = None,
    ):
        assert return_info
        
        if self.mjx_env.done("game"):
            self.curr_obs_dict = self.mjx_env.reset()

        # skip other players' turns
        while self.target_player not in self.curr_obs_dict:
            action_dict = {
                player_id: self.opponen_agents[player_id].act(obs)
                for player_id, obs in self.curr_obs_dict.items()
            }
            self.curr_obs_dict = self.mjx_env.step(action_dict)
            # game ends without player_0's turn
            if self.mjx_env.done("game"):
                self.curr_obs_dict = self.mjx_env.reset()

        assert self.target_player in self.curr_obs_dict
        obs = self.curr_obs_dict[self.target_player]
        feature_small = obs.to_features(feature_name="mjx-small-v0")
        feature_han22 = obs.to_features(feature_name="han22-v0")
        feature_han22 = np.delete(feature_han22, [0, 3], 0) # duplicate features
        feat = np.concatenate([feature_small, feature_han22], axis=0)
        mask = obs.action_mask()
        return feat, {"action_mask": mask}

    def step(self, action: int):
        # prepare action_dict
        action_dict = {}
        legal_actions = self.curr_obs_dict[self.target_player].legal_actions()
        action_dict[self.target_player] = mjx.Action.select_from(action, legal_actions)
        for player_id, obs in self.curr_obs_dict.items():
            if player_id == self.target_player:
                continue
            action_dict[player_id] = self.opponen_agents[player_id].act(obs)

        # update curr_obs_dict
        self.curr_obs_dict = self.mjx_env.step(action_dict)

        # skip other players' turns
        while self.target_player not in self.curr_obs_dict:
            action_dict = {
                player_id: self.opponen_agents[player_id].act(obs)
                for player_id, obs in self.curr_obs_dict.items()
            }
            self.curr_obs_dict = self.mjx_env.step(action_dict)

        # parepare return
        assert self.target_player in self.curr_obs_dict, self.curr_obs_dict.items()
        obs = self.curr_obs_dict[self.target_player]
        done = self.mjx_env.done(self.done_type)
        r = self.mjx_env.rewards(self.reward_type)[self.target_player]
        feature_small = obs.to_features(feature_name="mjx-small-v0")
        feature_han22 = obs.to_features(feature_name="han22-v0")
        feature_han22 = np.delete(feature_han22, [0, 3], 0) # duplicate features
        feat = np.concatenate([feature_small, feature_han22], axis=0)
        mask = obs.action_mask()

        return feat, r, done, {"action_mask": mask}

class REINFORCE:
    def __init__(self, model: nn.Module, opt: optim.Optimizer) -> None:
        self.model = model
        self.log_probs = 0
        self.entropy = 0
        self.opt: optim.Optimizer = opt

    def act(self, observation, action_mask):
        observation = torch.from_numpy(observation).flatten().float()
        mask = torch.from_numpy(action_mask)
        logits = self.model(observation)
        logits -= (1 - mask) * 1e9
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.entropy += dist.entropy()
        self.log_probs += log_prob

        assert action_mask[action.item()] == 1, action_mask[action.item()]
        return int(action.item())

    def update_gradient(self, R):
        self.opt.zero_grad()
        loss = -R * self.log_probs# - self.entropy * 0.001

        loss.backward()
        self.opt.step()

        self.log_probs = 0
        self.entropy = 0

oppponent = MLPAgent()

env = GymEnv(
    opponent_agents=[oppponent, oppponent, oppponent],
    reward_type="game_tenhou_7dan",
    done_type="game",
    # done_type="round",
    feature_type="mjx-small-v0", # not use
)

# setup
model = PolicyNetwork()
checkpoint = torch.load('./supervised_trained_model.pth')
model.load_state_dict(checkpoint)
opt = optim.Adam(model.parameters(), lr=1e-5)
agent = REINFORCE(model, opt)

file_path = './reinforce_model.pth'
resume = False

if resume == True:
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {file_path}")

def save_model(episode):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict()},
                file_path)

# train
loop_num = 100_000
avg_R = 0.0

for i in tqdm(range(loop_num)):
    if i > 0 and i % 10 == 0:
        print("log: ", i, avg_R/i, flush=True)
    if i > 0 and i % 100 == 0:
        save_model(i)

    obs, info = env.reset()

    done = False
    R = 0
    
    while not done:
        a = agent.act(obs, info["action_mask"])
        obs, r, done, info = env.step(a)
        R += r
    agent.update_gradient(R)
    avg_R += R

save_model(loop_num)