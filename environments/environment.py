import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs.array_spec import ArraySpec, BoundedArraySpec
from tf_agents.trajectories import time_step as ts


class ClassifyEnv(PyEnvironment):
    """
    In this custom `PyEnvironment` environment we define observations and rewards for the actions for learning a policy
    solving imbalanced classification.
    Based on https://www.tensorflow.org/agents/tutorials/2_environments_tutorial
    """

    def __init__(self, X_train, y_train, imb_ratio):
        """
        Initialization of the environment
        """
        self._action_spec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name="action")
        self._observation_spec = ArraySpec(shape=X_train.shape[1:], dtype=X_train.dtype, name="observation")
        self._episode_ended = False

        self.X_train = X_train
        self.y_train = y_train
        self.imb_ratio = imb_ratio  # Imbalance ratio

        self.X_len = self.X_train.shape[0]
        self.sample_ids = np.arange(self.X_len)  # IDs for the samples in X_train

        self.current_step = 0  # current step

        sample = self.sample_ids[self.current_step]  # the sample corresponding to 'self.current_step'
        self._state = self.X_train[sample]

    def action_spec(self):
        """Definition of the actions."""
        return self._action_spec

    def observation_spec(self):
        """Definition of the observations."""
        return self._observation_spec

    def _reset(self):
        """For every new training epoch we shuffle the data."""

        np.random.shuffle(self.sample_ids)

        self.current_step = 0  # Reset episode step counter at the end of every episode
        self._state = self.X_train[self.sample_ids[self.current_step]]
        self._episode_ended = False

        return ts.restart(self._state)

    def _step(self, action):
        """Take one step in the environment.
        If the action is correct, the environment will either return 1 or `imb_rate` depending on the current class.
        If the action is incorrect, the environment will either return -1 or -`imb_rate` depending on the current class.
        """
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start a new episode
            return self.reset()

        right_action = self.y_train[self.sample_ids[self.current_step]]
        self.current_step += 1

        if action == right_action:  # correct action
            if right_action:  # minority class
                reward = 1
            else:  # majority class
                reward = self.imb_ratio

        else:  # wrong action
            if right_action:  # minority class
                reward = -1
                self._episode_ended = True  # Stop if minority class is misclassified
            else:  # majority class
                reward = -self.imb_ratio

        if self.current_step == self.X_len - 1:  # If last step
            self._episode_ended = True

        self._state = self.X_train[self.sample_ids[self.current_step]]  # get next state

        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)
