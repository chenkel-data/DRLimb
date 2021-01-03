import os
from agents.dqn import DQNAgent
from data import *  # load_imdb

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

episodes = 2  # number of epochs to train DQN-Agent
replay_buffer_max_length = 100000  # size of replay memory
batch_size = 64
learning_rate = 1e-3
log_step = 50
eval_step = 50
gamma = 0.9  # Discount factor
epsilon = 0.1  # choosing random action
imb_ratio = 0.1  # Imbalance ratio

min_class = [2]  # Minority classes
maj_class = [0, 1, 3, 4, 5, 6, 7, 8, 9]  # Majority classes

X_train, y_train, X_test, y_test = load_image("mnist")

X_train, y_train, X_val, y_val, X_test, y_test = create_data(X_train, y_train, X_test, y_test, min_class, maj_class, imb_ratio=imb_ratio)

print('Distribution after imbalancing (training): {}'.format(Counter(y_train)))
print('Distribution after imbalancing (validation): {}'.format(Counter(y_val)))

collect_steps_per_episode = 50

conv_layers  = ((32, (5, 5), 2), (32, (5, 5), 2), )  # Convolutional layers
dense_layers = (256, 256,)  # Dense layers
dropout_layers = (0.2, 0.2,)  # Dropout layers
layers = {"conv": conv_layers, "dense": dense_layers, "dropout": dropout_layers}  # build a dict containing the underlying Q-Network Layers

model = DQNAgent()

model.compile(X_train, y_train, learning_rate, epsilon, gamma, imb_ratio, replay_buffer_max_length, layers)

model.fit(X_train, y_train, epochs=episodes, batch_size=batch_size, eval_step=eval_step, log_step=log_step,
          collect_steps_per_episode=collect_steps_per_episode)

model.evaluate(X_test, y_test, X_train, y_train)

