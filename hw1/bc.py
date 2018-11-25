import numpy as np, tensorflow as tf
import pickle as pkl
import argparse

def load_expert_data() -> dict:
	parser = argparse.ArgumentParser()
	parser.add_argument("data_path", type=str, nargs=None,
					help="pickle file path to rollout data")
	cmd_args = parser.parse_args()
	
	with open(cmd_args.data_path, 'rb') as f:
		data = pkl.load(f)
	return data
		
class Agent():
	def __init__(self, obs, actions):
		self.sess = tf.Session()
		self.obs = tf.placeholder([None] + obs.shape[1:], tf.float32)
		self.actions =  tf.placeholder([None] + actions.shape[1:], tf.float32)
		self.build_nn()

	def build_nn(self): 
		act_1 = tf.layers.Dense(128, activation=tf.nn.tanh)(self.obs)
		pred_actions = tf.layers.Dense(self.actions_shape, activation=None)(act_1)		
		self.loss = tf.sum(tf.reduce_mean(pred_actions - self.actions)**2)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

	def train(self, obs, actions):
		self.optimizer.minimize(self.loss)
		self.sess.run(self.optimizer, feed_dict= {self.obs:obs,
													self.actions:actions})

	def evaluate_agent():
		
if __name__ == "__main__":
	rollouts = load_expert_data()	
	bc_agent = Agent(rollouts['observations'], rollouts['actions'])
	
