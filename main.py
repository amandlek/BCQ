import gym
import numpy as np
import torch
import argparse
import os
import time
import datetime
import h5py

import utils
import DDPG
import BCQ

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    avg_success = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        obs = env._get_observation()
        done = False
        success = False
        # while not done:
        for _ in range(1000): 
            obs = np.concatenate([obs["robot-state"], obs["object-state"]])
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            success = success or env._check_success()
        avg_success += float(success)

    avg_reward /= eval_episodes
    return avg_reward, avg_success

def make_env(env_name):
    import robosuite
    env = robosuite.make(
        env_name,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        gripper_visualization=False,
        reward_shaping=True,
        control_freq=100,
    )
    return env

def setup(name):
    # tensorboard logging
    from tensorboardX import SummaryWriter
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')
    exp_dir = os.path.join(FILE_PATH, "./experiments/{}".format(name), time_str)
    os.makedirs(exp_dir)

    output_dir = os.path.join(FILE_PATH, "./trained_models/{}".format(name), time_str)
    os.makedirs(output_dir)

    return SummaryWriter(exp_dir), output_dir

def make_buffer(hdf5_path):
    """
    Add transition tuples from batch file to replay buffer.
    """
    rb = utils.ReplayBuffer()

    f = h5py.File(hdf5_path, "r")  
    demos = list(f["data"].keys())
    total_transitions = f["data"].attrs["total"]
    print("Loading {} transitions from {}...".format(total_transitions, hdf5_path))
    env_name = f["data"].attrs["env"]

    for i in range(len(demos)):
        ep = demos[i]
        obs = f["data/{}/obs".format(ep)][()]
        actions = f["data/{}/actions".format(ep)][()]
        rewards = f["data/{}/rewards".format(ep)][()]
        next_obs = f["data/{}/next_obs".format(ep)][()]
        dones = f["data/{}/dones".format(ep)][()]

        ### important: this is action clipping! ###
        actions = np.clip(actions, -1., 1.)

        zipped = zip(obs, actions, rewards, next_obs, dones)
        for item in zipped:
            ob, ac, rew, next_ob, done = item
            # Expects tuples of (state, next_state, action, reward, done)
            rb.add((ob, next_ob, ac, rew, done))
    f.close()

    return rb, env_name


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch")
    # parser.add_argument("--env_name", default="Hopper-v1")                # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    # parser.add_argument("--buffer_type", default="Robust")                # Prepends name to filename.
    parser.add_argument("--eval_freq", default=5e3, type=float)         # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)     # Max time steps to run environment for
    parser.add_argument("--name", default="test") # name of exp
    args = parser.parse_args()

    # if not os.path.exists("./results"):
    #     os.makedirs("./results")

    writer, output_dir = setup(args.name)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load buffer
    replay_buffer, env_name = make_buffer(args.batch)

    # env = gym.make(args.env_name)
    # env.seed(args.seed)
    env = make_env(env_name)
    
    obs = env.reset()
    obs = np.concatenate([obs["robot-state"], obs["object-state"]])

    state_dim = obs.shape[0] #env.observation_space.shape[0]
    action_dim = env.dof #env.action_space.shape[0] 
    max_action = 1. #float(env.action_space.high[0])

    # Initialize policy
    policy = BCQ.BCQ(state_dim, action_dim, max_action)

    episode_num = 0
    done = True 

    training_iters = 0
    num_epochs = 0
    last_time_saved = -1.
    best_success_rate = 0.
    while training_iters < args.max_timesteps: 
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq))
        avg_reward, avg_success = evaluate_policy(policy)

        ### TODO: TB avg return and success here ###
        for k in pol_vals:
            writer.add_scalar('BCQ/{}'.format(k), pol_vals[k], num_epochs)
        writer.add_scalar('BCQ/Rollout Return', avg_reward, num_epochs)
        writer.add_scalar('BCQ/Success Rate', avg_success, num_epochs)

        training_iters += args.eval_freq
        print("Training iterations: {}".format(training_iters))
        print("Epoch: {}".format(num_epochs))
        print("Losses: {}".format(pol_vals))
        print("Avg reward: {}, Avg Success: {}".format(avg_reward, avg_success))
        num_epochs += 1

        # save model every hour or when last record is beat
        if time.time() - last_time_saved > 3600:
            params_to_save = policy.get_dict_to_save()
            path_to_save = os.path.join(output_dir, "model_epoch_{}.pth".format(num_epochs))
            torch.save(params_to_save, path_to_save)

        if avg_success > best_success_rate:
            best_success_rate = avg_success
            params_to_save = policy.get_dict_to_save()
            path_to_save = os.path.join(output_dir, "model_epoch_{}_best_{}.pth".format(num_epochs, avg_success))
            torch.save(params_to_save, path_to_save)



