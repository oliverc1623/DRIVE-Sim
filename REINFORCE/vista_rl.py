import argparse
# from learn import Memory, Learner
# from learn_lstm import Learner
# from learn import Learner
from reinforce import Learner
# from learn_ppo import Learner 
# from learn_gpu_pairs import Learner
# from learn_gpu_trpo import Learner
# from learn_gpu_ppo import Learner

import matplotlib.pyplot as plt

def main(args):
    print(args.neuralnetwork)
    print(f"Animate: {args.animate}")
    learner = Learner(args.neuralnetwork, 
                    args.learning_rate, 
                    args.episodes, 
                    args.clip, 
                    args.animate,
                    args.algorithm,
                    args.filename)
    learner.learn()
    learner.save()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="VISTA Deep Reinforcement Learner")
    parser.add_argument("-nn", "--neuralnetwork")
    parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("-e", "--episodes", default=500, type=int)
    parser.add_argument("-c", "--clip", default=2, type=int)
    parser.add_argument("-a", "--animate", default=False, action='store_true', help='Bool type')
    parser.add_argument("--algorithm", default='vpg')
    parser.add_argument("-f", "--filename")
    args = parser.parse_args()
    main(args)

