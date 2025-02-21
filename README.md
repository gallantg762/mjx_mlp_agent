# About

simple mjx agent using mlp.

- install mjx `pip3 install git+https://github.com/mjx-project/mjx`
- 3-layer neural network
- Supervised learning with [houou rank paifu](https://tenhou.net/ranking.html)
- and Reinforcement Learning

I referred to god site https://note.com/oshizo/n/n61441adc340c.

running on [RiichiLab](https://mjai.app/users/gallantg762)

# Files

- mlp_agent.py
  - agent
- game_test.py
  - game simulater
- reinforce_model.pth
  - weight(with optimizer)
- rl_gym.py
  - reinforcement Learning
- tools/
  - some tools for make model

[large file on google drive](https://drive.google.com/drive/folders/17-yovA5bIAVRWeyn_UPS6PRXCeRYvfdO?usp=drive_link)

### In/Out Vector

- In
  - [mjx 107*34 2d vec mjx-small-v0](https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/include/mjx/internal/observation.cpp#L149) + [mjx 107*34 2d vec han22-v0](https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/include/mjx/internal/observation.cpp#L149) (remove dupulicated future)

```
all hand (0~3)
closed hand (4~7)
open hand (8~11)
Shanten information (12, 13)
last discarded tile (14)
last drawn tile (15)
? (16)
red in closed hand (17)
open events (18~44)
discard events (45~69)
dora (70~77)
bakaze (78)
jikaze (79)
latest event (80)
legal actions(81~106)
```

- out
  - [mjx 0~180 uint8_t](https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/include/mjx/internal/action.h#L45-L61)

```
0~33: Discard m1~rd
34,35,36: Discard m5(red), p5(red), s5(red)
37~70: Tsumogiri m1~rd
71,72,73: Tsumogiri m5(red), p5(red), s5(red)
74~94: Chi m1m2m3 ~ s7s8s9
95,96,97: Chi m3m4m5(red), m4m5(red)m6, m5(red)m6m7
98,99,100: Chi p3p4p5(red), p4p5(red)p6, p5(red)p6p7
101,102,103: Chi s3s4s5(red), s4s5(red)s6, s5(red)s6s7
104~137: Pon m1~rd
138,139,140: Pon m5(w/ red), s5(w/ red), p5(w/ red)
141~174: Kan m1~rd
175: Tsumo
176: Ron
177: Riichi
178: Kyuushu
179: No
180: Dummy
```
