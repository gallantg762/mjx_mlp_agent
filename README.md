# Note

base on [mjx](https://github.com/mjx-project/mjx) specific commit([fcdac0eabf854c2a530168eda989479f41681ef9](https://github.com/mjx-project/mjx/commit/fcdac0eabf854c2a530168eda989479f41681ef9)).

clone and `make dist && python3 setup.py install`.

**But build is broken**. fix manually.

# About

simple mjx agent using mlp.

Just learned 3-layer neural network with Houou rank paifu from Tenhou .

After 5,000 matches against 3 `RuleBasedAgents` (mjx embedded agents), the average rank is 1.63.

I referred to god site https://note.com/oshizo/n/n61441adc340c.

# Files

- mlp_agent.py
  - agent
- game_test.py
  - game simulater
- model.pth
  - weight
- tools/
  - some tools for make model

### In/Out Vector

- In
  - [mjx 107*34 2d vec mjx-small-v0](https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/include/mjx/internal/observation.cpp#L149) + [mjx 107*34 2d vec han22-v0](https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/include/mjx/internal/observation.cpp#L149) (remove dupulicated future)

- out
  - [mjx 0~180 uint8_t](https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/include/mjx/internal/action.h#L45-L61)


