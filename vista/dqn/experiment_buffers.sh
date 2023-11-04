#!/bin/bash

printf "Running experiment buffers.\nComparing memory replay buffers to prioritized replay buffers in DQN for cartpole-v1.\n"

printf "running mem_dqn.py\n"
python mem_dqn.py

printf "running per_dqn.py"
python per_dqn.py
