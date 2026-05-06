# 🏃 Cliff Walking RL Demo — Q-learning vs SARSA

> **🌐 Live Demo (GitHub Pages):** https://leolung88.github.io/DRL_HW2/

本專案為深度強化學習課程作業，在經典的 **Cliff Walking（懸崖行走）** 環境中實作並比較兩種表格型強化學習演算法：
- **Q-learning**（Off-policy，離策略）
- **SARSA**（On-policy，同策略）

透過相同的超參數設定進行訓練，分析兩者在學習行為、策略選擇與穩定性上的差異。

---

## 📁 專案結構

```
HW2/
├── index.html                  # 互動式前端（GitHub Pages，純 JS + Chart.js）
├── run_cliff_walking.py        # Python 核心：環境 + 演算法 + 訓練 + 視覺化
├── app.py                      # Streamlit 版本（本機執行用）
├── report.md                   # Task 5：完整比較分析報告
├── qlearning_vs_sarsa_intro.md # 演算法介紹（含應用場景）
├── homework_overview.md        # 原始作業需求文件
├── need_to_do.md               # 任務拆解與邊界設定
└── learning_curve.png          # 訓練曲線圖（run_cliff_walking.py 產出）
```

---

## 🚀 執行方式

### 方法一：直接開啟瀏覽器 Demo（無需安裝）
點擊上方的 **Live Demo** 連結，即可在瀏覽器中即時調整超參數並執行訓練，完全不需要後端伺服器。

### 方法二：本機執行 Python 腳本
```bash
# 安裝依賴
pip install numpy matplotlib

# 執行訓練（產出訓練曲線圖 learning_curve.png）
python run_cliff_walking.py
```

### 方法三：本機執行 Streamlit 應用
```bash
# 安裝依賴
pip install streamlit numpy pandas

# 啟動互動介面
streamlit run app.py
```

---

## ⚙️ 實驗參數

| 參數 | 值 |
|:--|:--:|
| 環境 | Cliff Walking (4 × 12) |
| 學習率 α | 0.1 |
| 折扣因子 γ | 0.9 |
| 探索率 ε | 0.1（ε-greedy） |
| 訓練回合數 | 500 |
| 隨機種子 | 42 |

---

## 📊 實驗結果摘要

| 指標 | Q-learning (Off-policy) | SARSA (On-policy) |
|:--|:--:|:--:|
| 最後 50 回合平均獎勵 | -61.50 | **-17.78** |
| 最終路徑長度（ε=0） | **13 步** | 15 步 |
| 訓練穩定性 | 低（持續震盪） | **高**（~100 回穩定） |
| 路徑策略 | 緊貼懸崖（冒險） | 繞行上層（安全） |

### Q-learning 最終路徑（Off-policy）
```
. . . . . . . . . . . .
. . . . . . . . . . . .
x x x x x x x x x x x x   ← 緊貼懸崖上方（最短路徑）
S C C C C C C C C C C G
```

### SARSA 最終路徑（On-policy）
```
. . . . . . . . . . . .
x x x x x x x x x x x x   ← 遠離懸崖一層（安全路徑）
x . . . . . . . . . . x
S C C C C C C C C C C G
```

圖例：`S`=起點、`G`=終點、`C`=懸崖、`x`=行走路徑

> 詳細分析請參閱 [report.md](./report.md)

---

## 🔑 核心發現

1. **SARSA 的訓練期表現優於 Q-learning**：因為它的策略本身就避開了懸崖，在 ε-greedy 探索期間掉崖次數遠少於 Q-learning。

2. **Q-learning 學到理論最優路徑**：在 ε=0 的純貪婪部署下，Q-learning 找到 13 步的最短路徑，是全局最優解。

3. **On-policy vs Off-policy 的本質差異**：Q-learning 假設未來每步都完美（Bellman Optimality），SARSA 將探索的隨機性納入評估（Bellman Expectation），兩者在有「風險懸崖」的環境中會收斂到截然不同的策略。

---

## 📚 參考文獻

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. — Chapter 6.5: Sarsa: On-policy TD Control
