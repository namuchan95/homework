import numpy as np, tensorflow as tf
import pickle as pkl
import argparse

def load_expert_data(data_path:str) -> dict:
	with open(data_path, 'rb') as f:
		data = pkl.load(f)
	return data
		
class Agent():
	def __init__(self, env, obs_size, actions_size):
		self.sess = tf.Session()
		self.env = gym.make(env)	
		self.obs = tf.placeholder([None] + list(self.env.observation_space.shape), tf.float32)
		self.actions =  tf.placeholder([None] + list(self.env.action_space.shape), tf.float32)
		self.build_nn()

	def build_nn(self): 
		act_1 = tf.layers.Dense(128, activation=tf.nn.tanh)(self.obs)
		pred_actions = tf.layers.Dense(self.actions_shape, activation=None)(act_1)
		self.loss = tf.sum(tf.reduce_mean(pred_actions - self.actions)**2)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

	def train(self, obs, actions):#need to set the batch size and GD steps
		self.optimizer.minimize(self.loss)
		self.sess.run(self.optimizer, feed_dict= {self.obs:obs,
													self.actions:actions})

	def evaluate_agent(self, num_rollouts):
		def rollout(self):
			obs = self.env.reset()
			done = False
			total_return = 0
			while not done:
				action = self.predict(obs)
				obs, ret, done, _ = self.env.step(action)
				total_return += ret
			return total_return
		
		env_stats=[]
		for _ in range(num_rollouts):
			env_stats += [self.rollout()]
		return np.mean(env_stats), np.var(env_stats)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("data_path", type=str, nargs=None,
					help="pickle file path to rollout data")
	cmd_args = parser.parse_args()
	env_name = cmd_args.data_path.split('/')[1].split('.')[0]
	rollouts = load_expert_data(cmd_args.env)	
	bc_agent = Agent(env_name)
	
