### Note

I use mjx specific commit https://github.com/mjx-project/mjx/commit/fcdac0eabf854c2a530168eda989479f41681ef9.

clone and `python3 setup.py install`.

**But build is broken**, fix manually.

### About

simple mjx agent using mlp.

Just learned 3-layer neural network with 30,000 games of Houou rank paifu from Tenhou .

After 5,000 matches against 3 `RuleBasedAgents` (mjx embedded agents), the average rank is 1.88.

I referred to god site https://note.com/oshizo/n/n61441adc340c.

### Curious

I want to know how strong this.

Beat this AI and let me know.

### Files

- mlp_agent.py
  - agent
- game_test.py
  - game simulater
- model_tenhou_mlp_30000.pth
  - weight
- tools/
  - some tools for make model
