from collections import defaultdict
import numpy as np

class Qlearning():
    def __init__(self, env, alpha, gamma, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(float)
        self.n_actions = env.action_space.n

    def choose_action(self, state):
        # epsilon greedy working:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # exploring 
        else:
            return self.get_best_action(state) # exploit
        
    def get_best_action(self, state):
        bestQ = -float('inf')
        best_action = 0
        for a in range(self.n_actions):
            q_val = self.Q[(state, a)]
            if q_val > bestQ:
                bestQ = q_val
                best_action = a
        return best_action
    
    def update(self, state, action, reward, next_state, terminated, truncated):#next action?
        currentQ = self.Q[(state, action)]

        if terminated == True or truncated == True:
            TDtarget = reward
        else:        
            # ecuacion for updating Qlearning:
            best_q = max([self.Q[(next_state, a)] for a in range(self.n_actions)])
            TDtarget = reward + self.gamma*best_q
        self.Q[(state, action)] = (1-self.alpha)*currentQ + self.alpha*(TDtarget)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, max_ep = 3000):
        episode_reward = []
        episode_steps = []
        for ep in range(max_ep):
            state, info = self.env.reset()
            reward_sum = 0
            steps = 0
            terminated = False
            truncated = False

            while not terminated and not truncated:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.update(state, action, reward, next_state, terminated, truncated)
                # no need for a "next_action" valuable, QLearning always uses max Q (off policy)
                reward_sum += reward
                state = next_state
                steps += 1
            episode_reward.append(reward_sum)
            episode_steps.append(steps)
            self.decay_epsilon()
        return episode_reward, episode_steps
