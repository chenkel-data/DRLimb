### Imbalanced Classification with Deep Reinforcement Learning

An implementation of a Deep Q-network for imbalanced data classification based on the
 [paper](https://arxiv.org/pdf/1901.01379.pdf) by _Lin et al_.

_A  deep Q-learning  network  (DQN)  based  model  for  im-balanced  data  classification  is  proposed  in  this  paper.
In  our model,  the  imbalanced  classification  problem  is  regarded  asa guessing game which can be decomposed into 
a sequential decision-making process. At each time step, the agent receives an environment state which is represented 
by a training sample and  then  performs  a  classification  action  under  the  guidance of a policy. If the agent 
performs a correct classification action it will be given a positive reward, otherwise, it will be given a  negative  
reward.  The  reward  from  minority  class  is  higher than that of majority class. The goal of the agent is to obtain 
as more cumulative rewards as possible during the process of sequential decision-making, that is, to correctly recognize
the samples as much as possible._

#### Requirements
 * requirements.txt
 * For training on the creditcard fraud dataset, create a ./data folder in the root with the file _creditcard.csv_ from  [kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).