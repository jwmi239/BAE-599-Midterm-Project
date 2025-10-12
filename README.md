# 🏃‍♂️ Cardiovascular Fitness Prediction Using NHANES Data

A machine learning web application that predicts cardiovascular fitness levels using health metrics from the National Health and Nutrition Examination Survey (NHANES) dataset.

## 🎯 Project Overview

This project uses Random Forest classification to predict cardiovascular fitness levels (Low, Moderate, High) based on data from NHANES - a comprehensive health survey conducted by the CDC that combines interviews and physical examinations to assess health and nutritional status in the United States.

**Key Features**:
- VO2 Max (maximum oxygen consumption)
- Age 
- Gender
- Body Mass Index (BMI)

**Model Performance**: 99.43% accuracy on test data with minimal overfitting.

## 🚀 Live Demo

[View the Streamlit App](https://your-app-name.streamlit.app) *(Add your deployed URL here)*

## 📊 Dataset

- **Source**: NHANES (National Health and Nutrition Examination Survey) 1999-2004 cycles
- **Original Size**: 31,126 participants → 8,324 with VO2 Max → 4,403 complete cases
- **Final Sample**: 4,403 participants with complete health profiles
- **Age Range**: 18-50 years
- **Data Completeness**: 100% (no missing values in final dataset)

## 🔬 Key Features

### 📈 Interactive Web Application
- **Data Exploration**: Comprehensive visualizations and statistical analysis
- **Model Performance**: Detailed accuracy metrics and confusion matrices
- **Feature Importance**: Analysis of which factors most influence fitness predictions
- **Interactive Predictions**: Real-time fitness level prediction with custom inputs

### 🧠 Machine Learning Pipeline
- **Algorithm**: Random Forest Classifier (100 trees)
- **Preprocessing**: StandardScaler for continuous features
- **Validation**: Stratified train-test split (80/20)
- **Performance**: 99.43% test accuracy, >99% precision/recall across all classes

## 🛠️ Installation & Setup

### Local Development

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/cardiovascular-fitness-prediction.git
cd cardiovascular-fitness-prediction
```

2. **Install dependencies**:
```bash
pip install -r requirements_streamlit.txt
```

3. **Run the Streamlit app**:
```bash
streamlit run streamlit_app.py
```

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account

2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**

3. **Deploy the app** by connecting your GitHub repository

4. **Set the main file path**: `streamlit_app.py`

5. **Requirements file**: `requirements_streamlit.txt`

## 📁 Project Structure

```
├── streamlit_app.py                 # Main Streamlit application
├── requirements_streamlit.txt       # Python dependencies for deployment
├── NHANES_VO2Max_Complete_NoMissing_FINAL.csv  # Processed dataset
├── Calories Burned Predictor.ipynb # Original Jupyter notebook
├── NHANES Visualizations.ipynb     # Data visualization notebook
├── Combining Datasets.ipynb        # Data preprocessing notebook
├── ml_workflow.py                   # Standalone ML pipeline
└── README.md                       # This file
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 99.43% |
| **Training Time** | 127ms |
| **Overfitting** | 0.57% (minimal) |
| **Precision** | >99% (all classes) |
| **Recall** | >99% (all classes) |

### Feature Importance Rankings:
1. **VO2 Max**: 63.8% (Primary predictor)
2. **Age**: 20.5% (Secondary predictor) 
3. **Gender**: 11.7% (Demographic factor)
4. **BMI**: 3.9% (Minor predictor)

## 🔍 Application Sections

### 1. 🏠 Project Overview
- Research background and objectives
- Dataset summary and key statistics
- Fitness level distribution analysis

### 2. 📊 Data Inspection & Visualizations
- Box plots showing feature distributions by fitness level
- Gender distribution analysis across fitness categories
- Correlation matrix and relationship analysis
- Scatter plots revealing patterns in the data

### 3. 🧹 Data Cleaning Process
- Detailed documentation of all preprocessing steps
- Justifications for each cleaning decision
- Impact analysis on model performance
- Data quality metrics and retention rates

### 4. 🤖 ML Model Selection
- Comprehensive algorithm comparison
- Justification for Random Forest selection
- Training strategy and hyperparameter choices
- Cross-validation approach

### 5. 📈 Model Application
- Training process and efficiency metrics
- Detailed performance breakdown by class
- Confusion matrix visualization
- Feature importance analysis with explanations

### 6. 🎯 Results Interpretation
- Clinical and practical implications
- Model limitations and considerations
- Future enhancement opportunities
- Real-world application scenarios

### 7. 🔮 Interactive Predictions
- Real-time fitness level prediction
- Probability distributions for all classes
- Feature contribution analysis
- Comparison with dataset averages

## 🏥 Clinical Applications

This model could be applied in:
- **Clinical Settings**: Rapid fitness assessment when VO2 Max testing isn't available
- **Fitness Centers**: Initial client evaluation and goal setting
- **Public Health**: Population-level screening and intervention targeting
- **Research**: Standardized fitness classification for studies

## ⚠️ Limitations

- Limited to ages 18-50 years
- Based on NHANES population (may not represent all demographics)
- Temporal considerations (1999-2004 data)
- Could benefit from additional cardiovascular markers

## 🚀 Future Enhancements

- Integration of additional health metrics (blood pressure, activity levels)
- Advanced ensemble methods and deep learning approaches
- Real-time integration with wearable devices
- Mobile application development
- Longitudinal validation studies

## 📈 Performance Metrics

The model achieves exceptional performance across all fitness levels:

- **Low Fitness**: 100% precision, 98.7% recall
- **Moderate Fitness**: 99.2% precision, 100% recall  
- **High Fitness**: 99.6% precision, 99.6% recall

With only 5 misclassifications out of 881 test samples, this model demonstrates remarkable accuracy for cardiovascular fitness prediction.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite:

```
Cardiovascular Fitness Prediction Using NHANES Data
Machine Learning Approach with Random Forest Classification
Based on National Health and Nutrition Examination Survey 1999-2004
```

## 📞 Contact

For questions or collaboration opportunities, please reach out via GitHub issues or email.

---

**Built with ❤️ using Streamlit, scikit-learn, and NHANES data**