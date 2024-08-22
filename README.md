### Note

base on [mjx](https://github.com/mjx-project/mjx) specific commit([fcdac0eabf854c2a530168eda989479f41681ef9](https://github.com/mjx-project/mjx/commit/fcdac0eabf854c2a530168eda989479f41681ef9)).

clone and `make dist && python3 setup.py install`.

**But build is broken**. I got error, and fix manually.

```
..../boost-src/libs/container_hash/include/boost/container_hash/hash.hpp:130:33: error: no template named 'unary_function' in namespace 'std'; did you mean '__unary_function'?
        struct hash_base : std::unary_function<T, std::size_t> {};
                           ~~~~~^~~~~~~~~~~~~~
                                __unary_function
```

### About

simple mjx agent using mlp.

Just learned 3-layer neural network with 40,000 games of Houou rank paifu from Tenhou .

After 5,000 matches against 3 `RuleBasedAgents` (mjx embedded agents), the average rank is 1.79.

I referred to god site https://note.com/oshizo/n/n61441adc340c.

### Strength

test data is 1000 game of houou player.

test accuracy of all situation is 70.9 %.

and accuracy of discard situation is

|model|accuracy|
|--|--|
|this|64.3%|
|Suphx|76.7%|
|akochan|65.1%|

may be weak :(


### Curious

I want to know how strong this.

Beat this AI and let me know.

### Files

- mlp_agent.py
  - agent
- game_test.py
  - game simulater
- model_tenhou_mlp_*.pth
  - weight
- tools/
  - some tools for make model
