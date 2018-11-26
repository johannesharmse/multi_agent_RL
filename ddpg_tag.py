# credit https://github.com/rohan-sawhney/multi-agent-rl/blob/master/ddpg_tag.py

import gym
import numpy as np
import tensorflow as tf
import argparse
import itertools
import time
import os
import pickle
import code
import random

from ddpg import Actor, Critic
from memory import Memory
from ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from make_env import make_env
import general_utilities
import simple_tag_utilities


def play(episodes, is_render, is_testing, checkpoint_interval,
         weights_filename_prefix, csv_filename_prefix, batch_size):
    # init statistics. NOTE: simple tag specific!
    statistics_header = ["episode"]
    statistics_header.append("steps")
    statistics_header.extend(["reward_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["loss_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["collisions_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["ou_theta_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["ou_mu_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["ou_sigma_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["ou_dt_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["ou_x0_{}".format(i) for i in range(env.n)])
    print("Collecting statistics {}:".format(" ".join(statistics_header)))
    statistics = general_utilities.Time_Series_Statistics_Store(
        statistics_header)

    for episode in range(args.episodes):
        # start state for all agents
        states = env._reset()
        episode_losses = np.zeros(env.n)
        episode_rewards = np.zeros(env.n)
        collision_count = np.zeros(env.n)
        steps = 0

        # while within bounds
        while True:
            steps += 1

            # render
            if args.render:
                env._render(close=False)

            # act
            # for each agent and adversary, 
            # choose an action
            actions = []
            for i in range(env.n):
                # an action has two coordinates
                # it will move in both x and y direction
                # each action gets a bit of noise to 
                # promote exploration
                # however it gets clipped to not wander too far
                action = np.clip(
                    actors[i].choose_action(states[i]) + actors_noise[i](), -2, 2)
                # print(action)
                actions.append(action)

            # step
            # 
            # agent will take a step in both x and y direction
            # if both action coordinates is zero, 
            # it means the agent did not step
            states_next, rewards, done, info = env._step(actions)

            # learn
            # 
            # train mode
            if not args.testing:
                # determine indices within memory (duration of episode to consider to 
                # draw batch from) to draw as batch
                # 
                # if the number of episode timesteps stored is smaller than
                # batch size, take all the available timesteps as batch, 
                # but shuffle
                size = memories[0].pointer
                batch = random.sample(range(size), size) if size < batch_size else random.sample(
                    range(size), batch_size)

                # do training per agent
                for i in range(env.n):
                    # if agent dies/exits boundaries, it's bad...
                    if done[i]:
                        rewards[i] -= 500
                    
                    # store important variables for timestep for agent i
                    memories[i].remember(states[i], actions[i],
                                         rewards[i], states_next[i], done[i])

                    # if we have a memory which is at least 10 times 
                    # as long as our batch size, we can start learning, 
                    # otherwise their is too much dependence between sampled states
                    if memories[i].pointer > batch_size * 10:
                        # sample from agent memory for learning
                        s, a, r, sn, _ = memories[i].sample(batch)
                        # reshape for TF
                        r = np.reshape(r, (batch_size, 1))
                        # get critic loss and update critic
                        # based on actor ouput values
                        loss = critics[i].learn(s, a, r, sn)
                        # update actor weights
                        # actor learning seems to be non-optimal
                        # exploration is encouraged by adding random noise
                        # uses pure adam optimizer and SGD
                        actors[i].learn(s)
                        # store episode loss as critic loss
                        # if critic has no loss, it means we know 
                        # exactly what actions to take with a lot of confidence
                        episode_losses[i] += loss
                    else:
                        episode_losses[i] = -1
            
            # get next state of each agent
            states = states_next
            # cumulative actor rewards
            episode_rewards += rewards
            # count collisions
            collision_count += np.array(
                simple_tag_utilities.count_agent_collisions(env))

            # reset states if done (any agent goes out of bounds)
            if any(done):
                # average reward per step
                episode_rewards = episode_rewards / steps
                # average loss per step
                episode_losses = episode_losses / steps

                statistic = [episode]
                statistic.append(steps)
                statistic.extend([episode_rewards[i] for i in range(env.n)])
                statistic.extend([episode_losses[i] for i in range(env.n)])
                statistic.extend(collision_count.tolist())
                statistic.extend([actors_noise[i].theta for i in range(env.n)])
                statistic.extend([actors_noise[i].mu for i in range(env.n)])
                statistic.extend([actors_noise[i].sigma for i in range(env.n)])
                statistic.extend([actors_noise[i].dt for i in range(env.n)])
                statistic.extend([actors_noise[i].x0 for i in range(env.n)])
                statistics.add_statistics(statistic)
                if episode % 25 == 0:
                    print(statistics.summarize_last())
                break

        if episode % checkpoint_interval == 0:
            statistics.dump("{}_{}.csv".format(
                csv_filename_prefix, episode))
            if not os.path.exists(weights_filename_prefix):
                os.makedirs(weights_filename_prefix)
            save_path = saver.save(session, os.path.join(
                weights_filename_prefix, "models"), global_step=episode)
            print("saving model to {}".format(save_path))
            if episode >= checkpoint_interval:
                os.remove("{}_{}.csv".format(csv_filename_prefix,
                                             episode - checkpoint_interval))

    return statistics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag_guided', type=str)
    parser.add_argument('--video_dir', default='videos/', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--episodes', default=100000, type=int)
    parser.add_argument('--video_interval', default=1000, type=int)
    parser.add_argument('--render', default=False, action="store_true")
    parser.add_argument('--benchmark', default=False, action="store_true")
    parser.add_argument('--experiment_prefix', default=".",
                        help="directory to store all experiment data")
    parser.add_argument('--weights_filename_prefix', default='/save/tag-ddpg',
                        help="where to store/load network weights")
    parser.add_argument('--csv_filename_prefix', default='/save/statistics-ddpg',
                        help="where to store statistics")
    parser.add_argument('--checkpoint_frequency', default=500, type=int,
                        help="how often to checkpoint")
    parser.add_argument('--testing', default=False, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument('--load_weights_from_file', default='',
                        help="where to load network weights")
    parser.add_argument('--random_seed', default=2, type=int)
    parser.add_argument('--memory_size', default=10000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--ou_mus', nargs='+', type=float,
                        help="OrnsteinUhlenbeckActionNoise mus for each action for each agent")
    parser.add_argument('--ou_sigma', nargs='+', type=float,
                        help="OrnsteinUhlenbeckActionNoise sigma for each agent")
    parser.add_argument('--ou_theta', nargs='+', type=float,
                        help="OrnsteinUhlenbeckActionNoise theta for each agent")
    parser.add_argument('--ou_dt', nargs='+', type=float,
                        help="OrnsteinUhlenbeckActionNoise dt for each agent")
    parser.add_argument('--ou_x0', nargs='+', type=float,
                        help="OrnsteinUhlenbeckActionNoise x0 for each agent")

    args = parser.parse_args()

    # store input arguments
    general_utilities.dump_dict_as_json(vars(args),
                                        args.experiment_prefix + "/save/run_parameters.json")

    # init env
    env = make_env(args.env, args.benchmark)

    # Extract ou initialization values
    if args.ou_mus is not None:
        if len(args.ou_mus) == sum([env.action_space[i].n for i in range(env.n)]):
            ou_mus = []
            prev_idx = 0
            for space in env.action_space:
                ou_mus.append(
                    np.array(args.ou_mus[prev_idx:prev_idx + space.n]))
                prev_idx = space.n
            print("Using ou_mus: {}".format(ou_mus))
        else:
            raise ValueError(
                "Must have enough ou_mus for all actions for all agents")
    else:
        ou_mus = [np.zeros(env.action_space[i].n) for i in range(env.n)]

    if args.ou_sigma is not None:
        if len(args.ou_sigma) == env.n:
            ou_sigma = args.ou_sigma
        else:
            raise ValueError("Must have enough ou_sigma for all agents")
    else:
        ou_sigma = [0.3 for i in range(env.n)]

    if args.ou_theta is not None:
        if len(args.ou_theta) == env.n:
            ou_theta = args.ou_theta
        else:
            raise ValueError("Must have enough ou_theta for all agents")
    else:
        ou_theta = [0.15 for i in range(env.n)]

    if args.ou_dt is not None:
        if len(args.ou_dt) == env.n:
            ou_dt = args.ou_dt
        else:
            raise ValueError("Must have enough ou_dt for all agents")
    else:
        ou_dt = [1e-2 for i in range(env.n)]

    if args.ou_x0 is not None:
        if len(args.ou_x0) == env.n:
            ou_x0 = args.ou_x0
        else:
            raise ValueError("Must have enough ou_x0 for all agents")
    else:
        ou_x0 = [None for i in range(env.n)]

    # if not os.path.exists(args.video_dir):
    #     os.makedirs(args.video_dir)
    # args.video_dir = os.path.join(
    #     args.video_dir, 'monitor-' + time.strftime("%y-%m-%d-%H-%M"))
    # if not os.path.exists(args.video_dir):
    #     os.makedirs(args.video_dir)
    # env = MyMonitor(env, args.video_dir,
    #                 # resume=True, write_upon_reset=True,
    #                 video_callable=lambda episode: (
    #                     episode + 1) % args.video_interval == 0,
    #                 force=True)

    # set random seed
    env.seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    # init actors and critics
    session = tf.Session()

    actors = []
    critics = []
    actors_noise = []
    memories = []

    # loop through agents (good agents and adversaries)
    for i in range(env.n):
        n_action = env.action_space[i].n
        state_size = env.observation_space[i].shape[0]
        # state is typically the position/distance of all other agents, 
        # adversaries and obstacles
        state = tf.placeholder(tf.float32, shape=[None, state_size])
        # agents are rewarded for closing in on adversaries, 
        # colliding with them, staying withing boundaries.
        # adversaries try to get away and avoid collisions
        reward = tf.placeholder(tf.float32, [None, 1])
        state_next = tf.placeholder(tf.float32, shape=[None, state_size])
        speed = 0.8 if env.agents[i].adversary else 1

        # actor network per agent
        actors.append(Actor('actor' + str(i), session, n_action, speed,
                            state, state_next))
        # critic network per agent
        critics.append(Critic('critic' + str(i), session, n_action,
                              actors[i].eval_actions, actors[i].target_actions,
                              state, state_next, reward))
        # actor network weights depend on critic network 
        # which determines the advantage of an action
        actors[i].add_gradients(critics[i].action_gradients)
        actors_noise.append(OrnsteinUhlenbeckActionNoise(
            mu=ou_mus[i],
            sigma=ou_sigma[i],
            theta=ou_theta[i],
            dt=ou_dt[i],
            x0=ou_x0[i]))
        memories.append(Memory(args.memory_size))

    # initialize all networks
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000000)

    # warm start if available
    if args.load_weights_from_file != "":
        saver.restore(session, args.load_weights_from_file)
        print("restoring from checkpoint {}".format(
            args.load_weights_from_file))

    start_time = time.time()

    # play
    statistics = play(args.episodes, args.render, args.testing,
                      args.checkpoint_frequency,
                      args.experiment_prefix + args.weights_filename_prefix,
                      args.experiment_prefix + args.csv_filename_prefix,
                      args.batch_size)

    # bookkeeping
    print("Finished {} episodes in {} seconds".format(args.episodes,
                                                      time.time() - start_time))
    tf.summary.FileWriter(args.experiment_prefix +
                          args.weights_filename_prefix, session.graph)
    save_path = saver.save(session, os.path.join(
        args.experiment_prefix + args.weights_filename_prefix, "models"), global_step=args.episodes)
    print("saving model to {}".format(save_path))
    statistics.dump(args.experiment_prefix + args.csv_filename_prefix + ".csv")