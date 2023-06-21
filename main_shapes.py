import logging

import os
import sys
import datetime
import argparse

import numpy as np

from agents.sfql import SFQL, SFTCoPQL
from features.tabular import TabularSF, TabularMVSF, TabularMOSF

from tasks.gridworld import MOShapes
from utils import stamp
from utils.config import parse_config_file
from utils.stats import OnlineMeanVariance

config_params = parse_config_file('gridworld.cfg')

gen_params = config_params['GENERAL']
task_params = config_params['TASK']
agent_params = config_params['AGENT']

rnsfql_params = config_params['RnSFQL']
rasfql_params = config_params['RaSFQL']
sftcopql_params = config_params['SFTCoPQL']

def load_tasks(trial_num):
    maze = np.array(task_params['maze'])
    weights = np.loadtxt('utils/shapes_weights.csv', delimiter=',')
    istart = gen_params['n_tasks'] * trial_num
    iend = istart + gen_params['n_tasks']
    weights = weights[istart:iend,:]
    assert weights.shape[0] == gen_params['n_tasks']
    rewards = [dict(zip(['1', '2', '3'], list(w.flatten()))) for w in weights]
    tasks = [MOShapes(maze=maze,
                      shape_rewards=rewards[t],
                      fail_prob=task_params['fail_prob'],
                      fail_reward=task_params['fail_reward'],
                      goal_reward=task_params['goal_reward']) for t in range(len(rewards))]
    return tasks

def train_agents(trial_num):
    agent_name = agent_params['agent_name']
    assert agent_name in ['rnsfql', 'rasfql', 'sftcopql']
    if agent_name == 'rnsfql':
        agent = SFQL(TabularSF(**rnsfql_params), **agent_params)
    elif agent_name == 'rasfql':
        agent = SFQL(TabularMVSF(**rasfql_params, risk_aversion=rasfql_params['penalty']), **agent_params)
    elif agent_name == 'sftcopql':
        agent = SFTCoPQL(TabularMOSF(**sftcopql_params), **agent_params)

    data_epis_return = OnlineMeanVariance()
    data_task_return = OnlineMeanVariance()
    data_cuml_return = OnlineMeanVariance()

    data_epis_failed = OnlineMeanVariance()
    data_task_failed = OnlineMeanVariance()
    data_cuml_failed = OnlineMeanVariance()

    data_r_good = OnlineMeanVariance()
    data_r_bad = OnlineMeanVariance()

    all_data = [data_epis_return, data_task_return, data_cuml_return, data_epis_failed, data_task_failed, data_cuml_failed, data_r_good, data_r_bad]
    data_names = ['epis_reward', 'task_reward', 'cuml_reward', 'epis_failed', 'task_failed', 'cuml_failed', 'data_r_good', 'data_r_bad']

    # training
    agent.reset(gen_params['n_tasks'], gen_params['n_samples'])
    tasks = load_tasks(trial_num)
    for itask, task in enumerate(tasks):
        agent_params['logger'].info('\n')
        agent.train_on_task(task, gen_params['n_samples'])

    # update performance statistics
    data_epis_return.update(np.column_stack([agent.episode_reward_hist_per_task]))
    data_task_return.update(np.column_stack([agent.reward_hist]))
    data_cuml_return.update(np.column_stack([agent.cum_reward_hist]))
    data_epis_failed.update(np.column_stack([agent.episode_fails_hist_per_task]))
    data_task_failed.update(np.column_stack([agent.fails_hist]))
    data_cuml_failed.update(np.column_stack([agent.cum_fails_hist]))
    data_r_good.update(np.column_stack([agent.r_good_hist]))
    data_r_bad.update(np.column_stack([agent.r_bad_hist]))

    # save mean performance
    label = 'shapes_{}_{}_{}_{}_'.format(agent.key, str(rasfql_params['penalty']).replace('.', ''), trial_num+1, stamp.get_timestamp())
    for data, data_name in zip(all_data, data_names):
        np.savetxt(agent_params['save_folder'] +"/"+ label + data_name + '.csv', data.mean, delimiter=',')

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--agent_name', type=str, default='sftcopql', choices=['rnsfql', 'rasfql', 'sftcopql'], help="Risk formulation of the agent.", required=True)
    parser.add_argument('--trial_num', type=int, default=1, help="1-30")
    parser.add_argument('--save_path', type=str, default='runs/gridworld', help='path to save results')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    # parse the args
    args = parse_option()

    agent_params['agent_name'] = args.agent_name

    now_str = datetime.datetime.now().strftime('%m%d_%H%M%S')
    save_folder = os.path.join(args.save_path, args.agent_name.upper(), now_str + "_" + str(args.trial_num))
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    logger = logging.getLogger(__name__)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")

    strHandler = logging.StreamHandler()
    strHandler.setFormatter(formatter)
    logger.addHandler(strHandler)
    logger.setLevel(logging.INFO)

    log_file = os.path.join(save_folder, 'print.log')
    log_fileHandler = logging.FileHandler(log_file)
    log_fileHandler.setFormatter(formatter)
    logger.addHandler(log_fileHandler)

    agent_params['save_folder'] = save_folder
    agent_params['logger'] = logger

    logger.info(args)
    train_agents(args.trial_num-1)
