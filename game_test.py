import mjx
from tqdm import tqdm
from mjx.agents import RuleBasedAgent
from mlp_agent import MLPAgent
from collections import Counter

agent_rule = RuleBasedAgent()
agent_mlp = MLPAgent()
agents = {
  'player_0': agent_mlp,
  'player_1': agent_rule,
  'player_2': agent_rule,
  'player_3': agent_rule
  }

# set round num
round_num = 10000

# for work
env = mjx.MjxEnv()
total_rank = 0
rank_counter = Counter()
rank_dict = {90: 1, 45: 2, 0: 3, -135: 4}
def print_stats(_round=round_num): print(f'round = {_round}', f'avg_rank = {round(total_rank / _round, 3)}', rank_counter)

# game loop
for i in tqdm(range(round_num)):
    obs_dict = env.reset()
    
    while not env.done():
        actions = {player_id: agents[player_id].act(obs) for player_id, obs in obs_dict.items()}
        obs_dict = env.step(actions)
    
    returns = env.rewards()
    reward = returns['player_0']
    rank_counter[reward] += 1
    total_rank += rank_dict[reward]

    if i>0 and i%(round_num/100)==0: print_stats(i)

print_stats()