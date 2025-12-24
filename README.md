# Twitter Sentiment Analysis
### Natural Language Processing • Artificial Neural Network • Streamlit • Sentiment Classification

Welcome to the **Twitter Sentiment Analysis** project, an end-to-end NLP application that predicts the sentiment of tweets using machine learning and deep learning techniques. This project includes **data preprocessing**, **model training** with both Logistic Regression and ANN, and a fully interactive **Streamlit web app** for real-time sentiment prediction.

---

## 📌 Table of Contents
- [🚀 Features](#-features)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🧠 Model Architecture](#-model-architecture)
- [📊 Workflow Overview](#-workflow-overview)
- [🧪 Sample Prediction Output](#-sample-prediction-output)
- [📈 Model Performance](#-model-performance)
- [🛠 Technologies Used](#-technologies-used)
- [💡 Future Enhancements](#-future-enhancements)
- [🤝 Contributing](#-contributing)
- [📬 Contact](#-contact)

---

## 🚀 Features

### ✔️ **1. Data Preprocessing**
- Text cleaning: Removing special characters, converting to lowercase.
- Stemming using Porter Stemmer.
- Stopword removal.
- TF-IDF vectorization for feature extraction.

### ✔️ **2. Machine Learning Models**
- Logistic Regression for baseline sentiment classification.
- Artificial Neural Network (ANN) using TensorFlow/Keras for improved accuracy.

### ✔️ **3. Interactive Streamlit Web Application**
- Clean UI for entering text/tweets.
- Real-time sentiment prediction with confidence scores.
- Responsive design.

### ✔️ **4. Dataset Handling**
- Uses the Sentiment140 dataset from Kaggle (1.6 million tweets).
- Balanced dataset with positive and negative sentiments.

### ✔️ **5. Model Persistence**
- Trained ANN model saved in Keras format.
- Preprocessed data saved as CSV for quick loading.

---

## 📁 Project Structure

```
SentimentAnalysis(Twitter)/
│
├── app.py                          # Streamlit web application
├── Twitter_Sentiment_Analysis_(NLP)_GFG.ipynb  # Jupyter notebook with full code
├── df_updated.csv                  # Preprocessed dataset
├── training.1600000.processed.noemoticon.csv  # Original dataset
├── twitter_ann_mode.keras          # Trained ANN model
├── kaggle.json                     # Kaggle API credentials
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore file
```

---

## ⚙️ Installation & Setup

### **1️⃣ Clone Repository**
```bash
git clone https://github.com/yourusername/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

### **2️⃣ Create Virtual Environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### **3️⃣ Install Required Packages**
```bash
pip install -r requirements.txt
```

### **4️⃣ Download Dataset (Optional)**
The dataset is already included, but if you want to download fresh:
- Place your `kaggle.json` in the root directory.
- Run the notebook to download and preprocess.

### **5️⃣ Run the Streamlit App**
```bash
streamlit run app.py
```

---

## 🧠 Model Architecture

### ANN Model:
```
Input: TF-IDF Vectorized Text (shape: [None, vocab_size])

Dense(128) → ReLU → Dropout(0.5)
Dense(64) → ReLU → Dropout(0.5)
Dense(1) → Sigmoid

Output: Probability of positive sentiment (0-1)
```

- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Metrics:** Accuracy

### Logistic Regression:
- Baseline model using TF-IDF features.
- Max iterations: 1000

---

## 📊 Workflow Overview

### **1. Data Acquisition**
- Download Sentiment140 dataset from Kaggle.
- Load and preprocess the CSV file.

### **2. Data Preprocessing**
- Rename columns for clarity.
- Convert target labels (4 → 1 for positive).
- Apply stemming and stopword removal.
- Save preprocessed data to `df_updated.csv`.

### **3. Feature Extraction**
- Split data into train/test (80/20).
- Fit TF-IDF vectorizer on training data.

### **4. Model Training**
- Train Logistic Regression and ANN models.
- Evaluate on test set.
- Save the best ANN model.

### **5. Deployment**
- Load model in Streamlit app.
- Preprocess user input and predict sentiment.

---

## 🧪 Sample Prediction Output

Example:

```
================ NEW PREDICTION REQUEST ================
Input Text: "I am so happy with this new product, it is amazing!"
-------------------------------------------------------
Sentiment: Positive
Confidence: 98.45%
😊 Positive Sentiment Detected!
========================================================
```

Another Example:

```
Input Text: "This is the worst experience ever, very disappointed."
-------------------------------------------------------
Sentiment: Negative
Confidence: 92.30%
😞 Negative Sentiment Detected!
========================================================
```

---

## 📈 Model Performance

| Model              | Accuracy | Notes                  |
| ------------------ | -------- | ---------------------- |
| Logistic Regression| ~78%    | Baseline model         |
| ANN (TensorFlow)   | ~82%    | Improved with dropout  |

*Note: Performance may vary based on dataset split and hyperparameters.*

---

## 🛠 Technologies Used

| Component            | Library/Technology         |
| -------------------- | -------------------------- |
| Programming Language | Python                     |
| Deep Learning        | TensorFlow, Keras         |
| Machine Learning     | Scikit-Learn              |
| NLP Processing       | NLTK                      |
| Web Framework        | Streamlit                 |
| Data Processing      | Pandas, NumPy             |
| Vectorization        | TF-IDF                    |

---

## 💡 Future Enhancements

- Add more advanced NLP techniques (e.g., BERT, LSTM).
- Implement multi-class sentiment (positive, negative, neutral).
- Add model explainability with SHAP or LIME.
- Deploy as a REST API using FastAPI.
- Integrate with real Twitter API for live tweet analysis.

---

## 🤝 Contributing

Contributions, pull requests, and suggestions are welcome!

To contribute:

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⭐ Support This Project

If this project helped you, please **star ⭐ the repository** — it encourages future improvements!

---

## 📬 Contact

For questions, suggestions, or collaboration:

**Email:** sm8939912@gmail.com

**GitHub:** [github.com/soumenmaity3](https://github.com/soumenmaity3)

---

*Made with ❤️ for the Data Science and NLP Community*