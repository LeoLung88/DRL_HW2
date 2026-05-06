# 深度強化學習作業 (HW2) - 待辦任務清單 (Tasks To Do)

這份文件基於 `homework_overview.md`，將作業拆解為數個具體的任務，並為每個任務設定了詳細的說明與邊界條件，以確保後續實作時的精準度並避免偏離目標（幻覺）。

## Task 1: 實作 Cliff Walking 網格世界環境 (Environment Implementation)
- **詳細說明**：
  - 建立一個 4 × 12 的二維矩陣（Gridworld）環境。
  - 定義起點（Start）在左下角 `(3, 0)`，終點（Goal）在右下角 `(3, 11)`。
  - 定義懸崖（Cliff）區域為底部的中間區段，即 `(3, 1)` 到 `(3, 10)`。
  - 實作環境互動的 `step(action)` 函數，接受上下左右四個動作。
  - 處理邊界碰撞（撞牆後停留在原地）。
- **邊界與限制 (Boundaries)**：
  - **嚴格遵守獎勵機制**：每走一步 reward = -1；掉入懸崖 reward = -100 且強制回到起點；到達終點即結束回合 (done = True)。
  - **不可以**增加額外的障礙物或改變網格大小。
  - **不可以**直接呼叫 OpenAI Gym / Gymnasium 內建的 CliffWalking 環境，應自行實作以確保細節完全符合題目要求及深入理解運作機制。

## Task 2: 實作 Q-learning 與 SARSA 演算法 (Algorithm Implementation)
- **詳細說明**：
  - 建立一個共用的 Q-table (狀態-動作價值函數) 資料結構，用於記錄每個座標 `(x, y)` 上四種動作的 Q 值，初始值可設為 0。
  - 實作 ε-greedy 策略函數，根據目前的 Q-table 與探索率選擇動作。
  - 實作 Q-learning (Off-policy) 的更新公式：`Q(S,A) ← Q(S,A) + α * [R + γ * max_a(Q(S',a)) - Q(S,A)]`
  - 實作 SARSA (On-policy) 的更新公式：`Q(S,A) ← Q(S,A) + α * [R + γ * Q(S',A') - Q(S,A)]`
- **邊界與限制 (Boundaries)**：
  - 此為**表格型強化學習 (Tabular RL)**，**嚴禁**使用任何深度神經網路（如 DQN 或 PyTorch/TensorFlow 架構）。
  - 兩者的 Q-table 初始化方式與資料結構必須完全一致。
  - 演算法的更新公式必須嚴格區分：Q-learning 用下一狀態的「最大 Q 值」進行更新；SARSA 必須根據 ε-greedy 實際選出的「下一個動作 A'」來進行更新。

## Task 3: 執行訓練過程 (Training Process)
- **詳細說明**：
  - 分別將 Q-learning 和 SARSA 實例化，並撰寫訓練迴圈。
  - 採用共用的超參數：例如 學習率 `α = 0.1`、折扣因子 `γ = 0.9`、探索率 `ε = 0.1`。
  - 訓練回合數 (Episodes) 至少設定為 500 回合。
  - 在每個回合中，記錄該回合獲得的「累積獎勵 (Total Reward)」。
- **邊界與限制 (Boundaries)**：
  - 兩種演算法的超參數與環境設定**必須完全相同**，以保證比較的絕對公平性。
  - 為了結果的可重現性，必須在訓練程式的開頭設定隨機數種子 (Random Seed)。
  - 除了題目規定的超參數外，不應隨意引入動態調整機制（如學習率衰減 learning rate decay 或 ε 衰減），除非題目允許或作為進階討論的對照組。

## Task 4: 實驗結果視覺化 (Result Visualization)
- **詳細說明**：
  - **累積獎勵曲線**：繪製一張折線圖，X 軸為回合數，Y 軸為累積獎勵，圖中需同時呈現 Q-learning 與 SARSA 的曲線以利直接對比。
  - **最終路徑視覺化**：在訓練結束後，以 `ε = 0` (完全貪婪策略) 跑一次測試回合，將 Q-learning 和 SARSA 最終學習到的最佳路徑繪製或印出在 4x12 的網格上。
- **邊界與限制 (Boundaries)**：
  - 圖表必須具備明確的圖例 (Legend)、X 軸與 Y 軸標籤以及標題。
  - 因早期訓練的 reward 可能為極小的負數（例如一直掉入懸崖），繪圖時需注意 Y 軸的範圍縮放，必要時可對曲線進行平滑化（但需保留原始數據走勢）。
  - 視覺化聚焦於「累積獎勵」與「學習路徑」，不要繪製過於複雜或無關的圖表。

## Task 4.5: 建立 Streamlit 互動式前端應用 (Web App Development)
- **詳細說明**：
  - 將前面實作的 Q-learning 與 SARSA 訓練過程、結果圖表以及 Cliff Walking 的最終路徑視覺化，封裝成一個 Streamlit 應用程式。
  - 介面需包含：超參數調整區（如學習率、回合數等）、啟動訓練按鈕、動態顯示訓練曲線的圖表區，以及呈現最終路徑的網格畫面。
  - 為了滿足「能透過 GitHub Demo (GitHub Pages) 運作」的嚴格限制，必須採用 **stlite** (Streamlit in WebAssembly) 技術。撰寫 `index.html` 載入 stlite 環境與 Python 腳本，使其成為一個能跑在瀏覽器裡的「純靜態網頁應用」。
- **邊界與限制 (Boundaries)**：
  - **嚴格遵守部署限制**：絕對不能依賴需要後端伺服器的部署方式。必須透過 stlite 轉為純前端架構，保證推送到 GitHub 且開啟 GitHub Pages 後即可直接運作。
  - 介面排版需能清楚並列或切換 Q-learning 與 SARSA 的結果以利比較。

## Task 5: 撰寫差異分析與總結報告 (Report Writing)
- **詳細說明**：
  - **學習表現**：根據 Task 4 的折線圖，比較兩者的收斂速度（誰先達到穩定狀態）。
  - **策略行為**：根據 Task 4 的路徑圖，分析兩者策略傾向（Q-learning 是否貼著懸崖走較冒險？SARSA 是否繞遠路較保守？）。
  - **穩定性分析**：比較兩者在訓練期間的表現波動程度，討論探索（exploration）對它們的影響為何不同。
  - **理論討論**：結合課堂理論，解釋為何 Off-policy (Q-learning) 與 On-policy (SARSA) 會在同樣的環境下學出截然不同的路線。
  - **結論要求**：給出明確的總結，點出收斂快慢、穩定性高低，以及未來在不同應用情境（允許失敗 vs 不允許失敗）下應如何選擇。
- **邊界與限制 (Boundaries)**：
  - 所有的分析與討論**必須基於 Task 3 和 Task 4 產出的實際實驗數據**，**嚴禁**憑空捏造結論或無視數據強行套用理論。
  - 理論說明必須符合標準強化學習文獻（如 Sutton & Barto 的 RL textbook）中對 Cliff Walking 環境的經典解釋，確保學理上的正確無誤。
