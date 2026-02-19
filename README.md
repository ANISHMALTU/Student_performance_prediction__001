# Student Performance Prediction - Project Overview

## 📊 Project Summary

This is a **Machine Learning-powered Web Application** that predicts student math scores based on various demographic and academic factors. The project combines data science with modern web technologies to provide an interactive platform for educational performance prediction.

---

## 🎯 Objective

To build a predictive model that can accurately predict a student's math score based on:
- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course
- Reading Score
- Writing Score

---

## 🏗️ Architecture

### Technology Stack

| Category | Technology |
|----------|------------|
| **Frontend** | HTML, CSS, JavaScript |
| **Backend** | Flask (Python Web Framework) |
| **Database** | SQLite |
| **ML Library** | Scikit-learn |
| **Other ML Models** | CatBoost, XGBoost |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |

### Project Structure

```
Student_Performance_Prediction/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── artifacts/                  # ML model files
│   ├── model.pkl              # Trained model
│   └── preprocessor.pkl       # Data preprocessor
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── utils.py
│   ├── exception.py
│   └── logger.py
├── notebook/
│   ├── 1. EDA STUDENT PERFORMANCE.ipynb
│   └── 2. MODEL TRAINING.ipynb
├── templates/                  # HTML templates
│   ├── index.html
│   ├── home.html
│   └── dashboard.html
└── static/                    # CSS files
```

---

## 🤖 Machine Learning Model

### Models Evaluated
1. Linear Regression ✅ (Selected)
2. Lasso Regression
3. Ridge Regression
4. K-Neighbors Regressor
5. Decision Tree
6. Random Forest Regressor
7. XGBoost Regressor
8. CatBoost Regressor
9. AdaBoost Regressor

### Data Preprocessing
- **Categorical Features**: OneHotEncoder
- **Numerical Features**: StandardScaler
- **Target Variable**: Math Score

### Performance Metrics
- R2 Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

## ✨ Key Features

### 1. **Prediction Interface**
   - User-friendly web form to input student data
   - Real-time prediction of math scores

### 2. **Dashboard**
   - View all historical predictions
   - Pagination support
   - Statistics overview (total predictions, average score, etc.)

### 3. **Advanced Statistics**
   - Gender-based analysis
   - Test preparation impact analysis
   - Parental education impact analysis
   - Lunch type correlation analysis
   - Score correlations

### 4. **Search & Filter**
   - Search predictions by various criteria
   - Filter by gender, ethnicity, test prep, score range

### 5. **Data Export**
   - Export all predictions to CSV format

### 6. **Trend Analysis**
   - Track predictions over time
   - Daily prediction statistics

### 7. **RESTful APIs**
   - `/api/predictions` - Get all predictions
   - `/api/statistics` - Get basic statistics
   - `/api/advanced-stats` - Get detailed statistics
   - `/api/trends` - Get trend data

---

## 🚀 How to Run

```
bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Access the app
# Open browser: http://localhost:5001
```

---

## 📈 Use Cases

1. **Educational Institutions**: Identify students who may need additional support
2. **Tutors**: Personalize teaching strategies based on predicted performance
3. **Researchers**: Analyze factors affecting student performance
4. **Parents**: Understand potential academic outcomes

---

## 🔮 Future Enhancements

- [ ] Add more sophisticated ML models (Deep Learning)
- [ ] Implement authentication system
- [ ] Add visualization dashboards (Charts/Graphs)
- [ ] Deploy to cloud platform
- [ ] Add mobile app support
- [ ] Include more prediction types (Science, English, etc.)

---

## 📝 Conclusion

This project demonstrates the end-to-end implementation of a machine learning application - from data exploration and model training to deploying a production-ready web application. It provides a solid foundation for educational analytics and can be extended for various use cases in the education sector.

---

**Created**: Student Performance Prediction Project
**Purpose**: ML Web Application for Educational Analytics
**Status**: ✅ Completed & Ready for Presentation
