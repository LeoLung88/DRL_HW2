"""
Streamlit 版本：Cliff Walking RL Demo
本機執行：streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Cliff Walking RL Demo", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Task 1: Cliff Walking 環境
# ──────────────────────────────────────────────────────────────────────────────
class CliffWalkingEnv:
    def __init__(self):
        self.rows, self.cols = 4, 12
        self.start_state, self.goal_state = (3, 0), (3, 11)
        self.state = self.start_state

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0: x = max(0, x - 1)
        elif action == 1: x = min(self.rows - 1, x + 1)
        elif action == 2: y = max(0, y - 1)
        elif action == 3: y = min(self.cols - 1, y + 1)
        self.state = (x, y)
        if x == 3 and 1 <= y <= 10:
            self.state = self.start_state
            return self.state, -100, False
        if self.state == self.goal_state:
            return self.state, -1, True
        return self.state, -1, False

# ──────────────────────────────────────────────────────────────────────────────
# Task 2: Agent + ε-greedy
# ──────────────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, rows=4, cols=12, n_actions=4, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_actions = n_actions
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.Q = np.zeros((rows, cols, n_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        x, y = state
        q = self.Q[x, y]
        return np.random.choice(np.flatnonzero(q == q.max()))


# ──────────────────────────────────────────────────────────────────────────────
# Task 3: 訓練函數
# ──────────────────────────────────────────────────────────────────────────────
def train_q_learning(env, episodes, alpha, gamma, epsilon):
    agent = Agent(env.rows, env.cols, 4, alpha, gamma, epsilon)
    rewards = []
    for _ in range(episodes):
        state, done, total = env.reset(), False, 0
        while not done:
            action = agent.choose_action(state)
            ns, reward, done = env.step(action)
            x, y = state; nx, ny = ns
            agent.Q[x, y, action] += alpha * (
                reward + gamma * np.max(agent.Q[nx, ny]) - agent.Q[x, y, action])
            state, total = ns, total + reward
        rewards.append(total)
    return agent, rewards

def train_sarsa(env, episodes, alpha, gamma, epsilon):
    agent = Agent(env.rows, env.cols, 4, alpha, gamma, epsilon)
    rewards = []
    for _ in range(episodes):
        state, done, total = env.reset(), False, 0
        action = agent.choose_action(state)
        while not done:
            ns, reward, done = env.step(action)
            na = agent.choose_action(ns)
            x, y = state; nx, ny = ns
            agent.Q[x, y, action] += alpha * (
                reward + gamma * agent.Q[nx, ny, na] - agent.Q[x, y, action])
            state, action, total = ns, na, total + reward
        rewards.append(total)
    return agent, rewards

def smooth(rewards, w=20):
    return pd.Series(rewards).rolling(window=w, min_periods=1).mean().values

def get_optimal_path(agent, env):
    orig = agent.epsilon
    agent.epsilon = 0.0
    state, done, path = env.reset(), False, [env.reset()]
    steps = 0
    while not done and steps < 100:
        action = agent.choose_action(state)
        state, _, done = env.step(action)
        path.append(state)
        steps += 1
    agent.epsilon = orig
    return path

def render_grid_html(path, env):
    """回傳 HTML table 表示路徑"""
    path_set = set(path)
    rows = []
    for r in range(env.rows):
        cells = []
        for c in range(env.cols):
            if (r, c) == env.start_state: emoji = "🏁"; bg = "#1a2a1a"
            elif (r, c) == env.goal_state: emoji = "🏆"; bg = "#2a1a1a"
            elif r == 3 and 1 <= c <= 10: emoji = "🌊"; bg = "#1e1010"
            elif (r, c) in path_set: emoji = "🐾"; bg = "#101e2a"
            else: emoji = "⬜"; bg = "transparent"
            cells.append(f'<td style="text-align:center;padding:4px;font-size:1.1rem;background:{bg};border:1px solid #2e3347;border-radius:4px;">{emoji}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table style="border-collapse:separate;border-spacing:2px;width:100%;">{"".join(rows)}</table>'


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("🏃 Cliff Walking RL Demo")
st.markdown("比較 **Q-learning（Off-policy）** 與 **SARSA（On-policy）** 在 Cliff Walking 環境中的學習行為差異。")

with st.sidebar:
    st.header("⚙️ 參數設定")
    episodes = st.slider("訓練回合數 (Episodes)", 100, 2000, 500, 100)
    alpha    = st.slider("學習率 α (Alpha)",     0.01, 1.0, 0.10, 0.01)
    gamma    = st.slider("折扣因子 γ (Gamma)",   0.10, 1.0, 0.90, 0.01)
    epsilon  = st.slider("探索率 ε (Epsilon)",   0.01, 1.0, 0.10, 0.01)
    run_btn  = st.button("🚀 開始訓練", use_container_width=True)

if run_btn:
    np.random.seed(42)
    env = CliffWalkingEnv()

    with st.spinner("訓練 Q-learning 中..."):
        q_agent, q_rewards = train_q_learning(CliffWalkingEnv(), episodes, alpha, gamma, epsilon)
    with st.spinner("訓練 SARSA 中..."):
        s_agent, s_rewards = train_sarsa(CliffWalkingEnv(), episodes, alpha, gamma, epsilon)

    st.success("✅ 訓練完成！")
    st.markdown("---")

    # ── 統計數字
    st.markdown("### 📊 統計摘要")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Q-learning 最後50回均值", f"{np.mean(q_rewards[-50:]):.1f}")
    c2.metric("SARSA 最後50回均值",      f"{np.mean(s_rewards[-50:]):.1f}")
    q_path = get_optimal_path(q_agent, CliffWalkingEnv())
    s_path = get_optimal_path(s_agent, CliffWalkingEnv())
    c3.metric("Q-learning 路徑長度", f"{len(q_path)-1} 步")
    c4.metric("SARSA 路徑長度",      f"{len(s_path)-1} 步")

    st.markdown("---")

    # ── 學習曲線 (Task 4)
    st.markdown("### 📈 學習曲線（平滑化）")
    chart_df = pd.DataFrame({
        "Q-learning (Off-policy)": smooth(q_rewards),
        "SARSA (On-policy)":       smooth(s_rewards),
    })
    st.line_chart(chart_df, height=300)

    st.markdown("---")

    # ── 路徑視覺化
    st.markdown("### 🗺️ 最終學習路徑（ε = 0 貪婪測試）")
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.markdown("#### 🟦 Q-learning（冒險短路）")
        st.markdown(render_grid_html(q_path, CliffWalkingEnv()), unsafe_allow_html=True)
    with gcol2:
        st.markdown("#### 🟥 SARSA（安全遠路）")
        st.markdown(render_grid_html(s_path, CliffWalkingEnv()), unsafe_allow_html=True)

else:
    st.info("👈 請在左側設定參數，並點擊「開始訓練」以啟動實驗。")
