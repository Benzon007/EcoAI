# 🌿 EcoAI – AI Climate Activist & Forecast Dashboard

### **An AI-driven environmental awareness platform that analyzes global temperature data, predicts climate trends, and explains causes, effects, and solutions for a sustainable future.**

---

## 🚀 Vision

EcoAI is an _AI Climate Activist_ — a system built to:

- Understand **why** climate patterns are changing
- Predict **how** they’ll evolve in the future
- Suggest **what actions** can be taken to mitigate harm

---

## 🧭 Development Phases

### 🧩 Phase 1: Data Understanding & Preprocessing

- Cleaned and structured global temperature data (1950–Present)
- Removed missing or inconsistent records
- Used `pandas` and `numpy` for preprocessing and data aggregation

### ⚙️ Phase 2: Predictive AI (LSTM Model)

- Built an **LSTM (Long Short-Term Memory)** model using **TensorFlow**
- Forecasted temperature trends for 6–60 months into the future
- Visualized results in a modern interactive dashboard

### 💻 Phase 3: Interactive Dashboard

- Developed a **Streamlit** app with:
  - Country dropdown (auto-populated)
  - Year range selector
  - AI forecast visualization
  - Summary metrics

### 🗄️ Phase 4: SQL Integration

- Migrated dataset into **SQLite** (`ecoai.db`)
- Used SQL queries for dynamic filtering
- Future scope: cloud-based PostgreSQL

### 🌍 Phase 5: EcoAI 2.0 – AI Climate Activist 🤖

_(Future Goal)_  
EcoAI will evolve into a conversational climate assistant that:

- Explains _why_ temperatures are changing
- Identifies _factors_ causing climate shifts
- Suggests _sustainability actions_
- Uses NLP + APIs (NASA, UNEP, World Bank) for real data insights

---

## 🧠 Architecture

```plaintext
CSV / API Data
      ↓
SQLite Database
      ↓
Preprocessing (Pandas)
      ↓
AI Model (TensorFlow)
      ↓
Visualization (Streamlit + Plotly)
      ↓
Future → NLP + Actionable Insights
```
