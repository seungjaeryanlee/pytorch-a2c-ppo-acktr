import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='Proximal Policy Optimization')

    # Environment
    parser.add_argument('--env-name', default='PongNoFrameskip-v4', dest='ENV_NAME',
                        help='environment to train on (default: PongNoFrameskip-v4)')

    # Optimizer arguments
    parser.add_argument('--lr', type=float, default=7e-4, dest='LR',
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False, dest='USE_LINEAR_LR_DECAY',
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--eps', type=float, default=1e-5, dest='EPS',
                        help='Adam optimizer epsilon (default: 1e-5)')

    # Deterministic
    parser.add_argument('--seed', type=int, default=1, dest='SEED',
                        help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False, dest='CUDA_DETERMINISTIC',
                        help="sets flags for determinism when using CUDA (potentially slow!)")

    # Hyperparameters - General
    parser.add_argument('--num-env-steps', type=int, default=10e6, dest='NUM_ENV_STEPS',
                        help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--gamma', type=float, default=0.99, dest='GAMMA',
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--num-processes', type=int, default=32, dest='NUM_PROCESSES',
                        help='how many training CPU processes to use (default: 32)')

    # Hyperparameters - GAE
    # TODO Should be True by default?
    parser.add_argument('--use-gae', action='store_true', default=False, dest='USE_GAE',
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95, dest='TAU',
                        help='gae parameter (default: 0.95)')

    # Hyperparameters - PPO
    parser.add_argument('--ppo-epoch', type=int, default=4, dest='PPO_EPOCH',
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32, dest='NUM_MINI_BATCH',
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.1, dest='CLIP_PARAM',
                        help='ppo clip parameter (default: 0.1)')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False, dest='USE_LINEAR_CLIP_DELAY',
                        help='use a linear schedule on the ppo clipping parameter')
    parser.add_argument('--entropy-coef', type=float, default=0.01, dest='ENTROPY_COEF',
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, dest='VALUE_LOSS_COEF',
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, dest='MAX_GRAD_NORM',
                        help='max norm of gradients (default: 0.5)')

    # Hyperparameters - Misc.
    # TODO What is this?
    parser.add_argument('--num-steps', type=int, default=5, dest='NUM_STEPS',
                        help='number of forward steps in A2C (default: 5)')

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
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    args.USE_CUDA = not args.no_cuda and torch.cuda.is_available()

    return args
