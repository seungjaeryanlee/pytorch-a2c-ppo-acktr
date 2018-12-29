#!/usr/bin/env python3
import copy
import glob
import os
import time
from collections import deque

import numpy as np
import torch
import wandb

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule


def add_log(j, total_num_steps, start, end, episode_rewards, dist_entropy, value_loss, action_loss):
    """
    Print log to console and add log to wandb.
    """
    print("Updates {}, num timesteps {}, FPS {}".format(
        j, total_num_steps, int(total_num_steps / (end - start))
    ))
    print("Last {} training episodes: ".format(len(episode_rewards)))
    print(" - mean/median reward {:.1f}/{:.1f}".format(
        np.mean(episode_rewards),
        np.median(episode_rewards),
    ))
    print(" - min/max reward {:.1f}/{:.1f}\n".format(
        np.min(episode_rewards),
        np.max(episode_rewards),
    ))
    print(' - dist_entropy {}'.format(dist_entropy))
    print(' - value_loss {}'.format(value_loss))
    print(' - action_loss {}'.format(action_loss))

    wandb.log({
        'FPS': int(total_num_steps / (end - start)),
        'Mean reward': np.mean(episode_rewards),
        'Median reward': np.median(episode_rewards),
        'Min reward': np.min(episode_rewards),
        'Max reward': np.max(episode_rewards),
        'dist_entropy': dist_entropy,
        'value_loss': value_loss,
        'action_loss': action_loss,
    }, step=j)


def clear_log_dirs(log_dir, eval_log_dir):
    """
    Remove old CSV files from previous code execution.

    Parameters
    ----------
    log_dir : str
        Directory where CSV log files are stored.
    eval_log_dir : str
        Directory where CSV log files for evaluation are stored.
    """
    # Delete old *.monitor.csv files in log_dir
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    # Delete old *.monitor.csv files in eval_log_dir
    try:
        os.makedirs(eval_log_dir)
    except OSError:
        files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def main():
    args = get_args()

    # Make results reproducible
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)
    if args.USE_CUDA and torch.cuda.is_available() and args.CUDA_DETERMINISTIC:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Cleanup log directories
    EVAL_LOG_DIR = args.LOG_DIR + "_eval"
    clear_log_dirs(args.LOG_DIR, EVAL_LOG_DIR)

    # 1 core per environment
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.USE_CUDA else "cpu")

    # Setup wandb
    wandb.init(project='ppo')
    wandb.config.update(args)

    # Setup Environment
    envs = make_vec_envs(args.ENV_NAME, args.SEED, args.NUM_PROCESSES,
                         args.GAMMA, args.LOG_DIR, args.add_timestep, device, False)
    obs = envs.reset()

    # Setup Agent
    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                          base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    agent = algo.PPO(actor_critic, args.CLIP_PARAM, args.PPO_EPOCH, args.NUM_MINI_BATCH,
                     args.VALUE_LOSS_COEF, args.ENTROPY_COEF, lr=args.LR,
                     eps=args.EPS, max_grad_norm=args.MAX_GRAD_NORM)
    rollouts = RolloutStorage(args.NUM_STEPS, args.NUM_PROCESSES,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    # TODO Magic number 10
    episode_rewards = deque(maxlen=10)
    NUM_UPDATES = int(args.NUM_ENV_STEPS) // args.NUM_STEPS // args.NUM_PROCESSES

    start = time.time()
    for j in range(NUM_UPDATES):

        # Decrease learning rate of optimizer linearly
        if args.USE_LINEAR_LR_DECAY:
            update_linear_schedule(agent.optimizer, j, NUM_UPDATES, args.LR)

        # TODO Why is CLIP_PARAM modified this way?
        # TODO Shouldn't this be conditioned on args.USE_LINEAR_CLIP_DELAY ?
        agent.CLIP_PARAM = args.CLIP_PARAM * (1 - j / float(NUM_UPDATES))

        for step in range(args.NUM_STEPS):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.USE_GAE, args.GAMMA, args.TAU)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.SAVE_INTERVAL == 0 or j == NUM_UPDATES - 1) and args.SAVE_DIR != "":
            save_path = os.path.join(args.SAVE_DIR, 'ppo')
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.USE_CUDA:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(save_path, args.ENV_NAME + ".pt"))

        total_num_steps = (j + 1) * args.NUM_PROCESSES * args.NUM_STEPS

        # Periodically print logs
        if j % args.LOG_INTERVAL == 0 and len(episode_rewards) > 1:
            end = time.time()
            add_log(j, total_num_steps, start, end, episode_rewards, dist_entropy, value_loss, action_loss)

        # Periodically evaluate
        if (args.EVAL_INTERVAL is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(
                args.ENV_NAME, args.SEED + args.NUM_PROCESSES, args.NUM_PROCESSES,
                args.GAMMA, EVAL_LOG_DIR, args.add_timestep, device, True)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.NUM_PROCESSES,
                                                       actor_critic.recurrent_hidden_state_size,
                                                       device=device)
            eval_masks = torch.zeros(args.NUM_PROCESSES, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)

                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                  format(len(eval_episode_rewards),
                         np.mean(eval_episode_rewards)))


if __name__ == "__main__":
    main()
