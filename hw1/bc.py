import numpy as np, tensorflow as tf
import pickle as pkl
import argparse
import gym

def load_expert_data(data_path):
	with open(data_path, 'rb') as f:
		data = pkl.load(f)
	return data
		
class Agent():
	def __init__(self, env):
		self.sess = tf.Session()
		self.env = gym.make(env)	
		self.obs = tf.placeholder(tf.float32, shape=tuple([None] + list(self.env.observation_space.shape)))
		self.actions =  tf.placeholder(tf.float32, shape= tuple([None] + list(self.env.action_space.shape)))
		self.build_nn()
		self.sess.run(tf.global_variables_initializer())

	def build_nn(self): 
		act_1 = tf.layers.Dense(128, activation=tf.nn.tanh)(self.obs)
		self.pred_actions = tf.layers.Dense(self.env.action_space.shape[0], activation=None)(act_1)
		self.loss = tf.reduce_sum(tf.reduce_mean((self.pred_actions - self.actions)**2, axis=0))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
		self.train_op = self.optimizer.minimize(self.loss)

	def train(self, obs, actions):#need to set the batch size and GD steps
		self.sess.run(self.train_op, feed_dict= {self.obs:obs,
													self.actions:actions})
	def predict(self, obs):
		return self.sess.run(self.pred_actions, feed_dict={self.obs : obs})

	def evaluate_agent(self, num_rollouts, render=True):
		def rollout():
			obs = self.env.reset()
			done = False
			total_return = 0
			while not done:
				action = self.predict(np.array([obs]))
				obs, ret, done, _ = self.env.step(action)
				if render : self.env.render()
				total_return += ret
			return total_return
		
		env_stats=[]
		for _ in range(num_rollouts):
			env_stats += [rollout()]
		return np.mean(env_stats), np.var(env_stats)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("data_path", type=str, nargs=None,
					help="pickle file path to rollout data")
	cmd_args = parser.parse_args()
	env_name = cmd_args.data_path.split('/')[1].split('.')[0]
	print("instantiating an object")
	rollouts = load_expert_data(cmd_args.data_path)	
	bc_agent = Agent(env_name)
	bc_agent.train(rollouts['observations'], rollouts['actions'].reshape(-1,8))
	print(bc_agent.evaluate_agent(100))
	
