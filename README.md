# Credit_risk_projects
Credit Risk PD Model Validation: XGBoost vs. Logistic Regression

Objective
To evaluate and compare the performance of XGBoost and Logistic Regression models for estimating Probability of Default (PD) in credit risk assessment.

Data & Preprocessing
The analysis uses historical credit data from LendingClub. Out of 75 original features, 22 specific variables were selected after dropping irrelevant operational columns (such as recoveries, total_rec_prncp, last_pymnt_amnt, out_prncp, id, member_id, url, and desc). Columns missing more than 50% of their values were excluded.

The binary target variable, loan_status, was constructed by classifying records with statuses of "Fully Paid" or "Does not meet the credit policy. Status:Fully Paid" as good loans (1). All other statuses were classified as bad loans (0).

Data Splitting
The dataset spans the years 2007 through 2014. It was partitioned using an out-of-time split approach: observations from 2007 to 2013 were allocated for model training, while data from 2014 served as the out-of-time test dataset.

Model Results & Evaluation
Model 1: XGBoost

Performance Metrics:
Train AUC: 0.8399 | Train Gini: 0.6799
Test AUC  : 0.6045 | Test Gini  : 0.2089

The Gini coefficient dropped significantly from 0.6799 on the training set to 0.2089 on the test set, indicating substantial overfitting.

Model 2: Logistic Regression

Performance Metrics:
LR Train AUC: 0.6569 | Gini: 0.3138
LR Test AUC  : 0.5770 | Gini: 0.1540

The low Gini coefficient of 0.3138 on the training set suggests potential underfitting, resulting in a test Gini of 0.1540.

Conclusion
Because both models yielded a test Gini coefficient below the 0.30 acceptance threshold, neither model is suitable for production in its current form and both are rejected.

Recommendations for Improvement
XGBoost: Mitigate overfitting by tuning hyperparameters—specifically by reducing maximum tree depth, increasing minimum child weight / leaf size, and applying regularisation.
   
Logistic Regression: Improve feature representations by performing feature scaling and applying Weight of Evidence (WoE) transformation to non-linear parameters prior to modeling.
