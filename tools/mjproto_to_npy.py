# * if using Apple Silicon, change arch
# env /usr/bin/arch -x86_64 /bin/zsh --login
#
# * DL paifu from https://tenhou.net/ranking.html
#
# * uncompress gz
# find ./ -type f -name "*.mjlog" -exec gunzip -S mjlog {} \;
# for filename in *.; do mv $filename ${filename%.}.mjlog; done
#
# * mjx-convert
# mjxc convert ./mjlog_dir ./mjxproto_dir --to-mjxproto
#
# * and convert npy
# python3 mjproto_to_npy.py

import numpy as np
from mjx import Observation, State, Action 
from tqdm import tqdm
import glob

# path to mjxproto
files = glob.glob("./mjxproto_dir/*")

# param
loop_limit = 10000
start_idx = 0
num_of_division = 100
out_dir = './npy'

# work variables
loop_cnt = 0
bar = tqdm(total = loop_limit)
bar.initial = start_idx
obs_hist = []
action_hist = []
file_cnt = start_idx // num_of_division

# convert loop
for file in files[start_idx:]:

    # save
    if loop_cnt % num_of_division == 0:
        if len(obs_hist) > 0 and len(action_hist) > 0:
            np.save(f"{out_dir}/tenhou_obs{file_cnt}.npy", np.stack(obs_hist))
            np.save(f"{out_dir}/tenhou_actions{file_cnt}.npy", np.array(action_hist, dtype=np.int32))
            obs_hist = []
            action_hist = []
            file_cnt += 1

    with open(file) as f:
        lines = f.readlines()

    bar.update(1)
    if loop_cnt >= loop_limit: break
    loop_cnt += 1

    for line in lines:
        state = State(line)

        for cpp_obs, cpp_act in state._cpp_obj.past_decisions():
            # to vector
            obs = Observation._from_cpp_obj(cpp_obs)
            featureS = obs.to_features(feature_name="mjx-small-v0")
            featureL = obs.to_features(feature_name="han22-v0")
            featureL_2 = np.delete(featureL, [0, 3], 0) # duplicate features
            feature = np.concatenate([featureS, featureL_2], axis=0)

            action = Action._from_cpp_obj(cpp_act)
            action_idx = action.to_idx()

            obs_hist.append(feature.ravel())
            action_hist.append(action_idx)