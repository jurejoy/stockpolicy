import numpy
import pandas as pd
import os
import random
import tensorflow as tf
tf.reset_default_graph()
MSFT_PATH = 'd:\\onedrive - microsoft\\ml\\stockenv'

def load_msft_data(MSFT_PATH):
    csv_path = os.path.join(MSFT_PATH, "MSFT.csv")
    return pd.read_csv(csv_path)

msft = load_msft_data(MSFT_PATH)

from numpy import genfromtxt
msft_array = genfromtxt('d:\\onedrive - microsoft\\ml\\stockenv\\msft.csv', delimiter=',')
prices = msft_array[:,1]
import numpy
prices = numpy.delete(prices,(0),axis=0)
#prices = numpy.linspace(1, 200, num=500)

'define a superclass for Decision policy'
class DecisionPolicy:
    def select_action(self, current_state, step):
        pass
    def update_q(self, state, action, reward, next_state):
        pass

'Inherit from supercalss to construct the random decision policy, the reture aciton will be [0,1,2]'
class RandomDecisionPolicy(DecisionPolicy):
    def __init__(self, actions):
        self.actions = actions
    def select_action(self, current_state, step):
        action = self.actions[random.randint(0, len(self.actions) - 1)]
        return action

def run_simulation(policy, initial_budget, initial_num_stocks, prices, hist, debug=False):
    budget = initial_budget
    num_stocks = initial_num_stocks
    share_value = 0
    transitions = list()
    '''
    # Save training model constant
    save_iterations = 100
    saver = tf.train.Saver()
'''


    # run i times iterations steps to go all the way through the prices matrix
    for i in range(len(prices) - hist -1):
        share_value = float(prices[i + hist +1])
        previous_share_value = float(prices[i + hist])
        current_state = numpy.asmatrix(numpy.hstack((prices[i:i+hist],budget, num_stocks)))
        current_portfolio = budget + num_stocks * previous_share_value
        


        if budget >= share_value and num_stocks > 0:
            action = policy.select_action(current_state, i)
            if action == 'Buy' and budget >= share_value:
                budget -= share_value
                num_stocks += 1
            elif action == 'Sell' and num_stocks > 0:
                budget += share_value
                num_stocks -= 1
            else:
                action = 'Hold'

            new_portfolio = budget + num_stocks * share_value
            reward = new_portfolio - current_portfolio
            next_state = numpy.asmatrix(numpy.hstack((prices[i+1:i+hist+1],budget, num_stocks)))
            transitions.append((current_state, action, reward, next_state))
            policy.update_q(current_state, action, reward, next_state)
        else:
            action = actions[random.randint(0, len(actions)-1)]
            if action == 'Buy' and budget >= share_value:
                budget -= share_value
                num_stocks += 1
            elif action == 'Sell' and num_stocks > 0:
                budget += share_value
                num_stocks -= 1
            else:
                action = 'Hold'

            new_portfolio = budget + num_stocks * share_value
            reward = new_portfolio - current_portfolio
            next_state = numpy.asmatrix(numpy.hstack((prices[i+1:i+hist+1],budget, num_stocks)))
            transitions.append((current_state, action, reward, next_state))

    
        if i % 100 == 0:
            print('progress {:.2f}%'.format(float(100 * i)/ (len(prices) - hist -1)))
            print(action)

        
    portfolio = budget + num_stocks * share_value
    print('Current budget is ${}, current share value is {} and own {} shares'.format(budget,share_value, num_stocks))
    if debug:
        print('${}\t{} shares'.format(budget, num_stocks))
    # load and save model
    policy.save_model()
    return portfolio


def run_simulations(policy, budget, num_stocks, prices, hist):
    num_tries = 1
    final_portfolios = list()
    for i in range(num_tries):
        final_portfolio = run_simulation(policy, budget, num_stocks, prices, hist)
        final_portfolios.append(final_portfolio)
        avg, std = numpy.mean(final_portfolios), numpy.std(final_portfolios)
    return avg, std


def verify_policy(policy, initial_budget, initial_num_stocks, prices, hist, debug=False):
    budget = initial_budget
    num_stocks = initial_num_stocks
    share_value = 0
    transitions = list()
    policy.restore_model()

    # run i times iterations steps to go all the way through the prices matrix
    for i in range(len(prices) - hist -1):
        share_value = float(prices[i + hist +1])
        previous_share_value = float(prices[i + hist])
        current_state = numpy.asmatrix(numpy.hstack((prices[i:i+hist],budget, num_stocks)))
        current_portfolio = budget + num_stocks * previous_share_value
        


        if budget >= share_value and num_stocks > 0:
            action = policy.select_action(current_state, i)
            if action == 'Buy' and budget >= share_value:
                budget -= share_value
                num_stocks += 1
            elif action == 'Sell' and num_stocks > 0:
                budget += share_value
                num_stocks -= 1
            else:
                action = 'Hold'

            new_portfolio = budget + num_stocks * share_value
            reward = new_portfolio - current_portfolio
            next_state = numpy.asmatrix(numpy.hstack((prices[i+1:i+hist+1],budget, num_stocks)))
            transitions.append((current_state, action, reward, next_state))

        else:
            action = actions[random.randint(0, len(actions)-1)]
            if action == 'Buy' and budget >= share_value:
                budget -= share_value
                num_stocks += 1
            elif action == 'Sell' and num_stocks > 0:
                budget += share_value
                num_stocks -= 1
            else:
                action = 'Hold'

            new_portfolio = budget + num_stocks * share_value
            reward = new_portfolio - current_portfolio
            next_state = numpy.asmatrix(numpy.hstack((prices[i+1:i+hist+1],budget, num_stocks)))
            transitions.append((current_state, action, reward, next_state))

        print('progress {:.2f}%'.format(float(100 * i)/ (len(prices) - hist -1)))
        print(action)

        
    portfolio = budget + num_stocks * share_value
    print('Current budget is ${}, current share value is {} and own {} shares'.format(budget,share_value, num_stocks))
    if debug:
        print('${}\t{} shares'.format(budget, num_stocks))
    return portfolio


'Reinforce learning part'

class QLearningDecisionPolicy(DecisionPolicy):
    def __init__(self, actions, input_dim):
        self.epsilon = 0.9
        self.gamma = 0.01
        self.actions = actions
        self.output_dim = len(actions)
        self.h1_dim = 5

        self.x = tf.placeholder(tf.float32, [None, input_dim])
        self.y = tf.placeholder(tf.float32, [self.output_dim])
        

        W1 = tf.Variable(tf.random_normal([input_dim, self.h1_dim]))
        b1 = tf.Variable(tf.constant(0.1, shape=[1, self.h1_dim]))
        h1 = tf.nn.leaky_relu(tf.matmul(self.x, W1) + b1)
        W2 = tf.Variable(tf.random_normal([self.h1_dim, self.output_dim]))
        b2 = tf.Variable(tf.constant(0.1, shape=[1, self.output_dim]))
        self.q = tf.nn.leaky_relu(tf.matmul(h1, W2) + b2)
        
    
        '''
        W1 = tf.Variable(tf.random_normal([input_dim, self.h1_dim]))
        b1 = tf.Variable(tf.constant(0.1, shape=[self.h1_dim]))
        h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
        W2 = tf.Variable(tf.random_normal([self.h1_dim, self.output_dim]))
        b2 = tf.Variable(tf.constant(0.1, shape=[self.output_dim]))
        self.q = tf.nn.relu(tf.matmul(h1, W2) + b2)
        '''

        loss = tf.square(self.y - self.q)
        self.train_op = tf.train.GradientDescentOptimizer(0.000001).minimize(loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()



    def select_action(self, current_state, step):
        #threshold = min(self.epsilon, step / 1000.)
        #action_q_vals = numpy.random.random((1,3))

        # Exploit best option with probability epsilon
        action_q_vals = self.sess.run(self.q, feed_dict={self.x:current_state})
        action_idx = numpy.argmax(action_q_vals)
        action = self.actions[action_idx]
        return action
        '''
        if random.random() < threshold:
            action_q_vals = self.sess.run(self.q, feed_dict={self.x:current_state})
            action_idx = numpy.argmax(action_q_vals)
            action = self.actions[action_idx]

        else:
        # Random option with probability 1 - epsilon
            action = self.actions[random.randint(0, len(self.actions)-1)]
        
        return action
        '''
    def update_q(self, state, action, reward, next_state):
        action_q_vals = self.sess.run(self.q, feed_dict={self.x:state})
        next_action_q_vals = self.sess.run(self.q, feed_dict={self.x:next_state})
        next_action_idx = numpy.argmax(next_action_q_vals)
        action_q_vals[0, next_action_idx] = reward  + self.gamma * next_action_q_vals[0, next_action_idx]
        action_q_vals = numpy.squeeze(numpy.asarray(action_q_vals))
        self.sess.run(self.train_op, feed_dict={self.x: state, self.y:action_q_vals})
    
    def close_session(self):
        self.sess.close()

    def restore_model(self):
        self.saver.restore(self.sess, "./stock.ckpt")

    def save_model(self):
        self.saver.save(self.sess, "./stock.ckpt")




# run simulation and train RL model
tf.reset_default_graph()
actions = ['Buy', 'Sell', 'Hold']
hist = 3
budget = 1000.0
num_stocks = 10
#policy = RandomDecisionPolicy(actions)
policy = QLearningDecisionPolicy(actions, 5)

result = run_simulation(policy,budget,num_stocks,prices, hist)
print("Current potofolio is ${}".format(result))
'''
avg,std = run_simulations(policy,budget,num_stocks,prices, hist)
policy.close_session
print(avg, std)
'''

#insert line
print('####################################################################')

# Verify Stock policy
prices = prices[len(prices)-20:len(prices)-2]
print(prices)

tf.reset_default_graph()
actions = ['Buy', 'Sell', 'Hold']
hist = 3
budget = 1000.0
num_stocks = 10

policy = QLearningDecisionPolicy(actions, 5)
result = verify_policy(policy,budget,num_stocks,prices, hist)
policy.close_session
print("Current potofolio is ${}".format(result))