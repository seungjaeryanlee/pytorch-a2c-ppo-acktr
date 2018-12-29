import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='Proximal Policy Optimization')

    # Environment
    parser.add_argument('--env-name', default='MontezumaRevengeNoFrameskip-v4', dest='ENV_NAME',
                        help='environment to train on (default: MontezumaRevengeNoFrameskip-v4)')

    # Optimizer arguments
    # Paper specifies learning rate of 0.0001
    parser.add_argument('--lr', type=float, default=1e-4, dest='LR',
                        help='learning rate (default: 1e-4)')
    # TODO Check whether paper uses this
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False, dest='USE_LINEAR_LR_DECAY',
                        help='use a linear schedule on the learning rate')
    # openai/random-network-distillation uses tf.train.AdamOptimizer with beta1=0.9, beta2=0.999, epsilon=1e-08
    # https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    # TODO Add beta1, beta2 for Adam
    parser.add_argument('--eps', type=float, default=1e-8, dest='EPS',
                        help='Adam optimizer epsilon (default: 1e-8)')

    # Deterministic
    parser.add_argument('--seed', type=int, default=1, dest='SEED',
                        help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False, dest='CUDA_DETERMINISTIC',
                        help="sets flags for determinism when using CUDA (potentially slow!)")

    # Hyperparameters - General
    # TODO Compute # of steps
    # 4.9152e8 = 128 env * 30000 rollouts per env * 128 steps per rollout
    # 1.2288e8 =  32 env * 30000 rollouts per env * 128 steps per rollout
    parser.add_argument('--num-env-steps', type=int, default=4.9152e8, dest='NUM_ENV_STEPS',
                        help='number of environment steps to train (default: 4.9152e8)')
    # Paper specifies external discount factor of 0.999
    # TODO External vs. Internal for RND
    parser.add_argument('--gamma', type=float, default=0.999, dest='GAMMA',
                        help='discount factor for rewards (default: 0.999)')
    # Paper specifies 128 parallel environments
    parser.add_argument('--num-processes', type=int, default=128, dest='NUM_PROCESSES',
                        help='how many training CPU processes to use (default: 128)')

    # Hyperparameters - GAE
    # Paper specifies using GAE
    parser.add_argument('--use-gae', action='store_true', default=True, dest='USE_GAE',
                        help='use generalized advantage estimation')
    # Paper specifies GAE parameter of 0.95
    # TODO Why not call it lambda?
    parser.add_argument('--tau', type=float, default=0.95, dest='TAU',
                        help='gae parameter (default: 0.95)')

    # Hyperparameters - PPO
    # Paper specifies optimization epoch of 4
    parser.add_argument('--ppo-epoch', type=int, default=4, dest='PPO_EPOCH',
                        help='number of ppo epochs (default: 4)')
    # Paper specifies 4 minibatches
    parser.add_argument('--num-mini-batch', type=int, default=4, dest='NUM_MINI_BATCH',
                        help='number of batches for ppo (default: 4)')
    # Paper specifies PPO clip parameter of 0.1
    parser.add_argument('--clip-param', type=float, default=0.1, dest='CLIP_PARAM',
                        help='ppo clip parameter (default: 0.1)')
    # TODO Check whether paper uses this
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False, dest='USE_LINEAR_CLIP_DELAY',
                        help='use a linear schedule on the ppo clipping parameter')
    # openai/random-network-distillation uses 0.001
    # https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/run_atari.py#L59
    parser.add_argument('--entropy-coef', type=float, default=0.001, dest='ENTROPY_COEF',
                        help='entropy term coefficient (default: 0.001)')
    # openai/random-network-distillation uses 1.0
    # https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/ppo_agent.py#L103
    parser.add_argument('--value-loss-coef', type=float, default=1.0, dest='VALUE_LOSS_COEF',
                        help='value loss coefficient (default: 1.0)')
    # openai/random-network-distillation does not use max_grad_norm
    # https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/run_atari.py#L128
    # TODO Implement disabling this
    parser.add_argument('--max-grad-norm', type=float, default=0.5, dest='MAX_GRAD_NORM',
                        help='max norm of gradients (default: 0.5)')

    # Hyperparameters - Misc.
    # TODO What is this? Is this rollout length?
    parser.add_argument('--num-steps', type=int, default=128, dest='NUM_STEPS',
                        help='number of forward steps in A2C (default: 128)')

    # Log / Save / Eval
    parser.add_argument('--log-dir', default='/tmp/gym/', dest='LOG_DIR',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--log-interval', type=int, default=10, dest='LOG_INTERVAL',
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-dir', default='./trained_models/', dest='SAVE_DIR',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--save-interval', type=int, default=100, dest='SAVE_INTERVAL',
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None, dest='EVAL_INTERVAL',
                        help='eval interval, one eval per n updates (default: None)')

    # Misc.
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    # Paper specifies CNN policy architecture (not using RNN)
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    args.USE_CUDA = not args.no_cuda and torch.cuda.is_available()

    return args
