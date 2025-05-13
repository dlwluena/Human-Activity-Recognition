
# Human Activity Recognition using KNN and PCA

This project implements a machine learning pipeline to classify human physical activities based on smartphone sensor data. It uses **K-Nearest Neighbors (KNN)** for classification and **Principal Component Analysis (PCA)** for dimensionality reduction.

---

## Project Aim

The goal is to predict one of the six activity classes —  
`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, and `LAYING` —  
based on 3-axis accelerometer and gyroscope signals collected from smartphones worn on the waist.

---

## Dataset

The dataset used is the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones), which contains:

- Time and frequency domain features from smartphone motion sensors
- 561 features per record
- Labels for six different human activities
- Data collected from 30 subjects aged 19-48

> The raw dataset is preprocessed and saved as:
- `hartrain.csv` (70% training set)
- `hartest.csv` (30% test set)

---

## Technologies & Libraries Used

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn (KNN, StandardScaler, PCA, GridSearchCV)
- Jupyter Notebook

---

## Pipeline Overview

1. **Load and inspect the dataset**
2. **Standardize features** using `StandardScaler`
3. **Reduce dimensionality** with `PCA`
4. **Train KNN classifier**
5. **Evaluate model performance**
6. **Optimize K** using `GridSearchCV`
7. **Visualize confusion matrix**

---

## How to Run

1. Download and unzip the UCI HAR Dataset.
2. Run the preprocessing script to generate `hartrain.csv` and `hartest.csv`.
3. Open `HAR_KNN_PCA_Project.ipynb` in Jupyter Notebook.
4. Execute each cell step-by-step to train and evaluate the model.

---

## Example Output

```
Best K: 7
Test Accuracy: 0.93
```

---

## References

- UCI Machine Learning Repository: [Human Activity Recognition](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)

---

## Note on pandas FutureWarning

If you see warnings like:
FutureWarning: The 'delim_whitespace' keyword in pd.read_csv is deprecated and will be removed in a future version.


Don't worry — this is **not an error**, just a deprecation notice from `pandas`.

### Recommended fix:
Replace `delim_whitespace=True` with `sep='\\s+'` in your `pd.read_csv()` calls.  
Both options behave similarly, but `sep='\\s+'` is the future-safe choice.

**Example:**
```python
# Old (deprecated)
pd.read_csv("file.txt", delim_whitespace=True)

# New (recommended)
pd.read_csv("file.txt", sep='\\s+')
```
