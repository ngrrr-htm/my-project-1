from QLearning import Qlearning
from SARSA import SARSA
from Expected_SARSA import Expected_SARSA
from mario_environment import mario
import sys
import matplotlib.pyplot as plt
import time 

env = mario()
agent_Q = Qlearning(env, alpha=0.5, gamma=0.9)
agent_S = SARSA(env, alpha=0.5, gamma=0.9)
agent_ES = Expected_SARSA(env, alpha=0.5, gamma=0.9)

# training:
reward_Q, steps_Q = agent_Q.train(max_ep=6000)
reward_S, steps_S = agent_S.train(max_ep=6000)
reward_ES, steps_ES = agent_ES.train(max_ep=6000)

# plot for rewards per episode
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(reward_Q, label='Q-learning')
plt.plot(reward_S, label='SARSA')
plt.plot(reward_ES, label='Expected SARSA')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Rewards per Episode')
plt.ylim(-200, 150)
plt.legend()
plt.tight_layout()
plt.show()

# plot for steps per episode 
plt.plot(steps_Q, label='Q-learning')
plt.plot(steps_S, label='SARSA')
plt.plot(steps_ES, label='Expected SARSA')
plt.xlabel('Episode')
plt.ylabel('Steps per Episode')
plt.title('Steps per Episode')
plt.ylim(0, 35)  
plt.legend()
plt.show()

# test:
'''Q Learning had the best results, so we'll be testing this algorithm: '''
agent_Q.epsilon = 0
test_episode = 10
result_per_episode = []

for ep in range(test_episode):
    state, _ = env.reset()
    done = False
    reward_total = 0
    steps_num = 0
    collected_coins = 0
    path = [state]
    ep_start = time.perf_counter()
    success = False

    while not done:
        action = agent_Q.get_best_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated # using OR rules
        reward_total += reward
        steps_num += 1
        path.append(next_state)
        if reward == 9: # 10 - 1 = 9
            collected_coins += 1
        state = next_state

    if state == env.gate:
        success = True
    ep_end = time.perf_counter()
    ep_time = ep_end - ep_start
    result_per_episode.append({'episode': ep + 1,'total_reward': reward_total,'steps': steps_num,'coins': collected_coins, 
                               'success': 'yes' if success == True else 'no', 'path': path, 'time': ep_time,
                               'end_status': 'Goal reached!' if success == True else 'Failed or max steps'})
    

'''Visualizing render: '''
env.render()  # Show initial grid
print()
while not done:
    action = agent_Q.get_best_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    total_reward += reward
    steps += 1
    
    print(f"Step {steps}: Action={action}, Reward={reward}")
    env.render()
    print()
    
    state = next_state
