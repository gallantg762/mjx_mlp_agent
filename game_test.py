import mjx
from tqdm import tqdm
from mjx.agents import RandomAgent, ShantenAgent, TsumogiriAgent, RuleBasedAgent
from mlp_agent import MLPAgent
from collections import Counter

# agent_rand = RandomAgent()
# agent_tsumogiri = TsumogiriAgent()
# agent_shanten = ShantenAgent()
agent_rule = RuleBasedAgent()
agent_mlp = MLPAgent()
agents = {
  'player_0': agent_mlp,
  'player_1': agent_rule,
  'player_2': agent_rule,
  'player_3': agent_rule
  }

# set round num
round = 1000

# for work
env = mjx.MjxEnv()
total_rank = 0
rank_counter = Counter()
rank_dict = {90: 1, 45: 2, 0: 3, -135: 4}
def print_stats(_round=round): print(f'round = {_round}', f'avg_rank = {total_rank / _round}', rank_counter)

# game loop
for i in tqdm(range(round)):
    obs_dict = env.reset()
    
    while not env.done():
        actions = {player_id: agents[player_id].act(obs) for player_id, obs in obs_dict.items()}
        obs_dict = env.step(actions)
    
    returns = env.rewards()
    reward = returns['player_0']
    rank_counter[reward] += 1
    total_rank += rank_dict[reward]

    if i>0 and i%(round/10)==0: print_stats(i)

print_stats()