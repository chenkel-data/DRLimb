import numpy as np
import tensorflow as tf
from environments.environment import ClassifyEnv
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers.tf_uniform_replay_buffer import \
    TFUniformReplayBuffer
from tf_agents.utils import common
from tf_agents.trajectories import trajectory

from sklearn.metrics import (confusion_matrix,
                             f1_score)

from metrics import custom_metrics


class DQNAgent:
    def __init__(self) -> None:
        """
        A class for training a TF-agent
        based on https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
        """

        self.train_env = None  # Training environment
        self.agent = None  # The algorithm used to solve an RL problem is represented by a TF-Agent
        self.replay_buffer = None  # The replay buffer keeps track of data collected from the environment
        self.dataset = None  # The agent needs access to the replay buffer via an iterable tf.data.Dataset
        self.iterator = None  # The iterator of self.dataset

    def compile(self, X_train: np.ndarray, y_train: np.ndarray, lr: float, epsilon: float, gamma: float, imb_ratio: float,
                replay_buffer_max_length: int, layers: dict) -> None:
        """
        Create the Q-network, agent and policy

        Args:
            X_train: A np.ndarray for training samples.
            y_train: A np.ndarray for the class labels of the training samples.
            lr: learn rate for the optimizer (default Adam)
            epsilon: Used for the default epsilon greedy policy for choosing a random action.
            gamma: The discount factor for learning Q-values
            imb_ratio: ratio of imbalance. Used to specifiy reward in the environment
            replay_buffer_max_length: Maximum lenght of replay memory.
            layers: A dict containing the layers of the Q-Network (eg, conv, dense, rnn, dropout).
        """

        dense_layers = layers.get("dense")
        conv_layers = layers.get("conv")
        dropout_layers = layers.get("dropout")

        self.train_env = TFPyEnvironment(ClassifyEnv(X_train, y_train, imb_ratio))  # create a custom environment

        q_net = QNetwork(self.train_env.observation_spec(), self.train_env.action_spec(), conv_layer_params=conv_layers,
                         fc_layer_params=dense_layers, dropout_layer_params=dropout_layers)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

        train_step_counter = tf.Variable(0)

        self.agent = DqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter,
            gamma=gamma,
            epsilon_greedy=epsilon,
        )

        self.agent.initialize()

        self.replay_buffer = TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=replay_buffer_max_length)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int, eval_step: int, log_step: int,
            collect_steps_per_episode: int) -> None:
        """
        Starts the training of the Agent.

        Args:
            X_train: A np.ndarray for training samples.
            y_train: A np.ndarray for the class labels of the training samples.
            epochs: Number of epochs to train Agent
            batch_size: The Batch Size
            eval_step: Evaluate Model each 'eval_step'
            log_step: Monitor results of model each 'log_step'
            collect_steps_per_episode: Collect a few steps using collect_policy and save to the replay buffer.
        """

        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(3)

        self.iterator = iter(self.dataset)

        def collect_step(environment, policy, buffer):
            time_step = environment.current_time_step()
            action_step = policy.action(time_step)
            next_time_step = environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)

            # Add trajectory to the replay buffer
            buffer.add_batch(traj)

        def collect_data(env, policy, buffer, steps):
            for _ in range(steps):
                collect_step(env, policy, buffer)

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        self.agent.train = common.function(self.agent.train)

        # Reset the train step
        self.agent.train_step_counter.assign(0)

        for _ in range(epochs):
            #print("epoch: ", _)
            # Collect a few steps using collect_policy and save to the replay buffer.
            collect_data(self.train_env, self.agent.collect_policy, self.replay_buffer, collect_steps_per_episode)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, _ = next(self.iterator)
            train_loss = self.agent.train(experience).loss

            step = self.agent.train_step_counter.numpy()

            if step % log_step == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % eval_step == 0:
                metrics = self.compute_metrics(X_train, y_train)
                print(metrics)

    def compute_metrics(self, X: np.ndarray, y_true: list) -> dict:
        """Compute Metrics for Evaluation"""
        # TODO: apply softmax layer for q logits?

        q, _ = self.agent._target_q_network (X, training=False)

        # y_scores = np.max(q.numpy(), axis=1)  # predicted scores (Q-Values)
        y_pred = np.argmax(q.numpy(), axis=1)  # predicted class label

        metrics = custom_metrics(y_true, y_pred)

        return metrics

    def evaluate(self, X: np.ndarray, y: list, X_train=None, y_train=None) -> dict:
        """
         Evaluation of trained Q-network
        """
        metrics = self.compute_metrics(X, y)

        print("evaluation: ", metrics)
        return metrics