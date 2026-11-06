# Smart Factory Alert Agent (Assignment 3)

智慧工廠設備異常警示 AI Agent：產生模擬感測器資料、清洗、規則式異常偵測、命令列警示輸出。Demo 截圖與技術報告位於專案外層的 `../report/` 資料夾。

## 專案結構

```text
assignment3/
├── generate_data.py    # 產生 300 筆模擬智慧工廠感測器 CSV 資料
├── data/sensor_data.csv # 工廠感測器資料（temp, pressure, vibration, label）
├── src/                # 專案源代碼
│   ├── preprocessing.py # 資料載入、缺值補值（中位數）、z-score 標準化
│   ├── detector.py     # 規則式異常偵測（閾值檢查 + 0-100 評分 + 嚴重度 + 建議）
│   └── agent.py        # CLI Agent：感知→推理→動作，輸出警示
├── main.py             # 專案入口：參數解析與 Agent 執行
├── tests/              # 單元測試（unittest），涵蓋前處理、偵測邊界、Agent 端到端
├── outputs/            # 執行結果輸出：alerts.csv (異常明細) + run_log.txt (執行記錄)
├── README.md           # 專案說明文件
└── requirements.txt    # 環境依賴：pandas>=2.0.0, numpy>=1.24.0
```

## 環境需求

- Python >= 3.9
- 依賴見 requirements.txt

## 安裝

```bash
pip install -r requirements.txt
```

## 使用方式

```bash
# 1. 產生資料（預設 300 筆、seed=42、異常率 10%、每欄缺值率 5%）
python generate_data.py

# 2. 執行 Agent
python main.py            # 預設 data/sensor_data.csv -> outputs/
python main.py --input data/sensor_data.csv --output outputs --log outputs/run_log.txt

# 3. 執行測試
python -m unittest discover -s tests
```

