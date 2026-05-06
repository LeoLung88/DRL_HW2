import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Task 1: 實作 Cliff Walking 網格世界環境
# ==========================================
class CliffWalkingEnv:
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start_state = (3, 0)
        self.goal_state = (3, 11)
        self.state = self.start_state
        
    def reset(self):
        """重置環境到起點"""
        self.state = self.start_state
        return self.state
        
    def step(self, action):
        """
        執行動作並返回 (next_state, reward, done)
        動作定義: 0: 上, 1: 下, 2: 左, 3: 右
        """
        x, y = self.state
        
        # 狀態轉移 (邊界碰撞處理)
        if action == 0:   # 上
            x = max(0, x - 1)
        elif action == 1: # 下
            x = min(self.rows - 1, x + 1)
        elif action == 2: # 左
            y = max(0, y - 1)
        elif action == 3: # 右
            y = min(self.cols - 1, y + 1)
            
        self.state = (x, y)
        
        # 判斷獎勵與是否結束
        if x == 3 and 1 <= y <= 10:
            # 掉入懸崖
            reward = -100
            self.state = self.start_state # 回到起點
            done = False # 回到起點繼續，通常懸崖不算是完成回合
        elif self.state == self.goal_state:
            # 到達終點
            reward = -1
            done = True
        else:
            # 正常移動
            reward = -1
            done = False
            
        return self.state, reward, done


# ==========================================
# Task 2: 實作 Q-learning 與 SARSA 演算法
# ==========================================
class Agent:
    def __init__(self, rows=4, cols=12, num_actions=4, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table 初始化為 0
        self.Q = np.zeros((rows, cols, num_actions))
        
    def choose_action(self, state):
        """ε-greedy 策略選擇動作"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions) # 探索
        else:
            x, y = state
            # 利用: 若有多個最大值，隨機選擇其中一個 (避免永遠只選第一個方向)
            q_values = self.Q[x, y, :]
            max_q = np.max(q_values)
            actions = np.where(q_values == max_q)[0]
            return np.random.choice(actions)

def q_learning(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    agent = Agent(env.rows, env.cols, 4, alpha, gamma, epsilon)
    rewards_history = []
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            # Q-learning 更新公式: Q(S,A) = Q(S,A) + alpha * [R + gamma * max(Q(S',a)) - Q(S,A)]
            x, y = state
            nx, ny = next_state
            best_next_action_q = np.max(agent.Q[nx, ny, :])
            agent.Q[x, y, action] += agent.alpha * (reward + agent.gamma * best_next_action_q - agent.Q[x, y, action])
            
            state = next_state
            total_reward += reward
            
        rewards_history.append(total_reward)
    return agent, rewards_history

def sarsa(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    agent = Agent(env.rows, env.cols, 4, alpha, gamma, epsilon)
    rewards_history = []
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        action = agent.choose_action(state)
        
        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.choose_action(next_state)
            
            # SARSA 更新公式: Q(S,A) = Q(S,A) + alpha * [R + gamma * Q(S',A') - Q(S,A)]
            x, y = state
            nx, ny = next_state
            agent.Q[x, y, action] += agent.alpha * (reward + agent.gamma * agent.Q[nx, ny, next_action] - agent.Q[x, y, action])
            
            state = next_state
            action = next_action
            total_reward += reward
            
        rewards_history.append(total_reward)
    return agent, rewards_history


# ==========================================
# Task 4: 實驗結果視覺化與路徑測試
# ==========================================
def get_optimal_path(agent, env):
    """將 epsilon 設為 0，取得完全貪婪的最佳路徑"""
    state = env.reset()
    path = [state]
    done = False
    
    # 暫存原本的 epsilon，設為 0
    temp_epsilon = agent.epsilon
    agent.epsilon = 0.0 
    
    steps = 0
    while not done and steps < 100: # 限制步數避免未收斂導致無限迴圈
        action = agent.choose_action(state)
        state, _, done = env.step(action)
        path.append(state)
        steps += 1
        
    # 恢復原本的 epsilon
    agent.epsilon = temp_epsilon 
    return path

def print_path_grid(path, env, title):
    """印出帶有路徑的網格"""
    grid = [['.' for _ in range(env.cols)] for _ in range(env.rows)]
    
    # 標示懸崖
    for y in range(1, env.cols - 1):
        grid[3][y] = 'C'
        
    # 標示起終點
    grid[3][0] = 'S'
    grid[3][env.cols - 1] = 'G'
    
    # 標示路徑
    for (x, y) in path:
        if grid[x][y] not in ['S', 'G', 'C']:
            grid[x][y] = 'x'
            
    print(f"\n=== {title} ===")
    for row in grid:
        print(" ".join(row))
    print(f"Path length: {len(path)-1} steps\n")


def smooth_rewards(rewards, window=10):
    """平滑化獎勵曲線以便觀察"""
    smoothed = np.zeros(len(rewards))
    for i in range(len(rewards)):
        start = max(0, i - window)
        smoothed[i] = np.mean(rewards[start:i+1])
    return smoothed


# ==========================================
# 主程式 (Task 3: 執行訓練過程)
# ==========================================
def main():
    # 固定隨機種子以求結果可重現性
    np.random.seed(42)
    
    env = CliffWalkingEnv()
    episodes = 500
    
    print("開始訓練 Q-learning...")
    q_agent, q_rewards = q_learning(env, episodes=episodes)
    
    print("開始訓練 SARSA...")
    sarsa_agent, sarsa_rewards = sarsa(env, episodes=episodes)
    
    print("訓練完成！\n")
    
    # 取得並列印最佳路徑
    q_path = get_optimal_path(q_agent, env)
    sarsa_path = get_optimal_path(sarsa_agent, env)
    
    print_path_grid(q_path, env, "Q-learning 最終路徑 (Off-policy)")
    print_path_grid(sarsa_path, env, "SARSA 最終路徑 (On-policy)")
    
    # 繪製累積獎勵曲線 (平滑化處理)
    q_rewards_smoothed = smooth_rewards(q_rewards, window=20)
    sarsa_rewards_smoothed = smooth_rewards(sarsa_rewards, window=20)
    
    plt.figure(figsize=(10, 6))
    # 繪製原始數據(半透明)
    plt.plot(q_rewards, alpha=0.3, color='blue', label='Q-learning (Raw)')
    plt.plot(sarsa_rewards, alpha=0.3, color='red', label='SARSA (Raw)')
    # 繪製平滑曲線
    plt.plot(q_rewards_smoothed, color='blue', linewidth=2, label='Q-learning (Smoothed)')
    plt.plot(sarsa_rewards_smoothed, color='red', linewidth=2, label='SARSA (Smoothed)')
    
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward per Episode')
    plt.title('Performance of Q-learning and SARSA on Cliff Walking')
    # 限制 Y 軸底部範圍，否則會被早期的 -100 拉得很難看
    plt.ylim(-200, 0)
    plt.legend()
    plt.grid(True)
    
    plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
    print("訓練曲線已儲存至 'learning_curve.png'")
    
    # 記錄最後幾回合的平均獎勵以供分析
    print(f"\n最後 50 回合平均獎勵:")
    print(f"Q-learning: {np.mean(q_rewards[-50:]):.2f}")
    print(f"SARSA:      {np.mean(sarsa_rewards[-50:]):.2f}")


if __name__ == "__main__":
    main()
