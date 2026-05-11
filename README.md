# 🏥 MedPredict AI — Multi-Disease Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006400?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

An AI-powered **multi-disease prediction system** that uses machine learning to predict the likelihood of **Heart Disease**, **Diabetes**, and **Breast Cancer** based on patient health parameters. Features a professional medical-themed dashboard with interactive visualizations.

> ⚠️ **Disclaimer:** This tool is for educational purposes only. Always consult a qualified medical professional.

---

## 📸 Screenshots

<!-- 
  TODO: Add screenshots of your app here!
  Take screenshots and save them in a 'screenshots' folder, then uncomment below:
  
  ![Dashboard](screenshots/dashboard.png)
  ![Heart Disease Prediction](screenshots/heart-prediction.png)
  ![Model Analytics](screenshots/model-analytics.png)
-->

*Screenshots coming soon — Run the app locally to see the full UI!*

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **3 Disease Models** | Heart Disease, Diabetes & Breast Cancer prediction |
| 🤖 **4 ML Algorithms** | Random Forest, Gradient Boosting, SVM & Logistic Regression |
| 📊 **Model Comparison** | Side-by-side accuracy, precision, recall, F1 & ROC-AUC |
| 📈 **Interactive Charts** | ROC curves, confusion matrices & feature importance (Plotly) |
| 🎯 **Real-time Prediction** | Enter patient data → get instant risk assessment with confidence % |
| 🎨 **Professional UI** | Custom medical-green theme with responsive layout |
| ⚡ **Cached Models** | Models trained once, cached for fast subsequent predictions |

## 📊 Model Performance

| Disease | Best Model | Accuracy | F1 Score |
|---------|-----------|----------|----------|
| Heart Disease | Random Forest | ~85% | ~84% |
| Diabetes | Gradient Boosting | ~79% | ~72% |
| Breast Cancer | Random Forest | ~97% | ~96% |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Core Language |
| **Streamlit** | Web Framework & Dashboard |
| **Scikit-Learn** | ML Model Training & Evaluation |
| **XGBoost** | Gradient Boosting Classifier |
| **Pandas & NumPy** | Data Processing & Analysis |
| **Plotly** | Interactive Visualizations |
| **Matplotlib** | Static Charts |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/kunchamarjun05/healthcare-ai.git
cd healthcare-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📁 Project Structure

```
healthcare-ai/
├── app.py              # Main Streamlit application (394 lines)
│                       # ├── Dashboard page with hero banner
│                       # ├── Heart Disease prediction UI
│                       # ├── Diabetes prediction UI
│                       # └── Breast Cancer detection UI
├── models.py           # ML model training & prediction logic
│                       # ├── Data loading functions
│                       # ├── train_and_evaluate() — trains 4 models
│                       # └── predict_single() — single patient prediction
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

## 🧠 How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│ Select       │ ──> │ Enter Patient │ ──> │ ML Model        │ ──> │ Risk         │
│ Disease      │     │ Parameters    │     │ Processes Data  │     │ Assessment   │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
                                                │
                                          ┌─────┴─────┐
                                          │ 4 Models   │
                                          │ Compared   │
                                          └───────────┘
```

1. **Select Disease** — Choose Heart Disease, Diabetes, or Breast Cancer
2. **Input Data** — Enter health parameters via interactive sliders
3. **Get Prediction** — Trained ML model returns risk level with confidence %
4. **View Analytics** — Explore ROC curves, confusion matrices & feature importance

## 🧪 Datasets Used

| Dataset | Source | Samples | Features |
|---------|--------|---------|----------|
| Cleveland Heart Disease | UCI ML Repository | 303 | 13 |
| Pima Indians Diabetes | Kaggle | 768 | 8 |
| Wisconsin Breast Cancer | Scikit-Learn | 569 | 30 |

## 📚 What I Learned

- End-to-end ML pipeline: data loading → preprocessing → training → evaluation → deployment
- Comparing multiple ML algorithms on the same dataset
- Building production-quality UIs with Streamlit
- Data visualization with Plotly (ROC curves, confusion matrices)
- Model caching for performance optimization

## 👨‍💻 Author

**Arjun Kuncham**  
🌐 [Portfolio](https://kunchamarjun05.github.io/portfolio/) • 💻 [GitHub](https://github.com/kunchamarjun05) • 📧 arjunkuncham05@gmail.com

---

⭐ Star this repo if you found it useful!
