### Note

base on [mjx](https://github.com/mjx-project/mjx) specific commit([fcdac0eabf854c2a530168eda989479f41681ef9](https://github.com/mjx-project/mjx/commit/fcdac0eabf854c2a530168eda989479f41681ef9)).

clone and `make dist && python3 setup.py install`.

**But build is broken**. fix manually.

### About

simple mjx agent using mlp.

Just learned 3-layer neural network with 50,000 games of Houou rank paifu from Tenhou .

After 5,000 matches against 3 `RuleBasedAgents` (mjx embedded agents), the average rank is 1.72.

I referred to god site https://note.com/oshizo/n/n61441adc340c.

### Strength

Match rate when discarding tiles in 1000 games of houou players.

|model|accuracy|
|--|--|
|this|64.8%|
|Suphx|76.7%|
|akochan|65.1%|

 <!-- 
100 matches against [mjsply](https://github.com/mjsplayer2020/mjsply), [manue](https://github.com/gimite/mjai-manue), average rank is

|model|avg rank|
|--|--|
|this||
|mjsply||
|manue||

※ mjsply's rate on [riichiLab](https://mjai.app/) is 1700 or above.
-->


### Curious

I want to know how strong this.

Beat this AI and let me know.

### Hope

I would like to enhance this using DQN.

### Files

- mlp_agent.py
  - agent
- game_test.py
  - game simulater
- model_tenhou_mlp_*.pth
  - weight
- tools/
  - some tools for make model

### In/Out Vector

In
[mjx 107*34 2d vec mjx-small-v0](https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/include/mjx/internal/observation.cpp#L149) + [mjx 107*34 2d vec han22-v0](https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/include/mjx/internal/observation.cpp#L149) (remove dupulicated future)

out
[mjx 180 1d vec](https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/include/mjx/internal/action.h#L45-L61)


