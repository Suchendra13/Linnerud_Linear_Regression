# Linear Regression on the Linnerud Dataset
This project demonstrates the implementation of a multi-output linear regression model using the Linnerud dataset from scikit-learn. The notebook covers the entire workflow from data loading and exploratory data analysis (EDA) to model training, evaluation, and visualization.

# Contents
1. [Dataset](#dataset)
2. [Project Workflow](#project-workflow)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Libraries Used](#libraries-used)
5. [How to Run](#how-to-run)

## Dataset
The project utilizes the `Linnerud dataset`, which is a multi-variate dataset containing physiological and exercise data for 20 middle-aged men.

* **Features (X - Exercise Data):**

1. Chins: Number of chin-ups
2. Situps: Number of sit-ups
3. Jumps: Number of jumping jacks

* **Targets (y - Physiological Data):**
1. Weight
2. Waist
3. Pulse

# ⚙️ Project Workflow
**Data Loading and Preparation:**

The Linnerud dataset is loaded from `sklearn.datasets`.

The features and targets are combined into a single pandas DataFrame for easier analysis.

The dataset is split into an 80% training set and a 20% testing set.

## Exploratory Data Analysis:

A statistical summary of the dataset is generated using .describe().

Data distributions are visualized using histograms and box plots.

Relationships and correlations between variables are explored using a pair plot and a correlation heatmap.

**Model Training:**

A LinearRegression model from scikit-learn is instantiated and trained on the training data (X_train, y_train).

**Model Evaluation:**

The trained model makes predictions on the test set (X_test).

The model's performance is evaluated using the following metrics:

**Mean Absolute Error (MAE):** 11.19

**Mean Squared Error (MSE):** 293.50

**Root Mean Squared Error (RMSE):** 17.13

**R-squared (R²):** -1.18

**Note:** The negative R-squared value indicates that the simple linear regression model performs poorly on this dataset, worse than a model that simply predicts the mean of the target values. This suggests the relationships are not well-captured by a linear model.

**Visualization:**

An Actual vs. Predicted scatter plot is created for the 'Weight' target to visually assess prediction accuracy.

A Residuals Plot is generated to check for patterns in the model's errors.

## 🛠️ Libraries Used
* `pandas`
* `numpy`
* `matplotlib.pyplot`
* `seaborn`
* `sklearn` (specifically `datasets`, `model_selection`, `linear_model`, `metrics`)
  
## How to Run
1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone [https://github.com/Suchendra13/Linnerud_Linear_Regression.git](https://github.com/Suchendra13/Linnerud_Linear_Regression.git)
    cd Linnerud_Linear_Regression
    ```
2.  **Ensure you have Jupyter Notebook installed** or use a compatible IDE (e.g., VS Code with Jupyter extensions).
3.  **Install the required Python libraries**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
4.  **Open the Jupyter Notebook**:
    ```bash
    jupyter notebook Linnerud_Linear_Regression.ipynb
    ```
5.  **Run all cells** in the notebook.
