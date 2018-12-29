import argparse
import os

import torch

from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize


# workaround to unpickle olf model files
import sys
sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1, dest='SEED',
                    help='random seed (default: 1)')
# TODO Not used?
parser.add_argument('--log-interval', type=int, default=10, dest='LOG_INTERVAL',
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4', dest='ENV_NAME',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/', dest='LOAD_DIR',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--add-timestep', action='store_true', default=False, dest='ADD_TIMESTEP',
                    help='add timestep to observations')
parser.add_argument('--non-det', action='store_true', default=False, dest='NON_DET',
                    help='whether to use a non-deterministic policy')
args = parser.parse_args()

# TODO Why reverse it?
args.det = not args.NON_DET

env = make_vec_envs(args.ENV_NAME, args.SEED + 1000, 1,
                    None, None, args.ADD_TIMESTEP, device='cpu',
                    allow_early_resets=False)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = \
    torch.load(os.path.join(args.LOAD_DIR, args.ENV_NAME + ".pt"))

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

if render_func is not None:
    render_func('human')

obs = env.reset()

if args.ENV_NAME.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)

    masks.fill_(0.0 if done else 1.0)

    if args.ENV_NAME.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    if render_func is not None:
        render_func('human')
