# PDAC-Texture
This project aims to develop predictive models for pancreatic ductal adenocarcinoma (PDAC) patients using radiomics features extracted from preoperative CT scans. Specifically, it addresses two key clinical objectives:

Pancreatic Texture Classification: Predicting intraoperative pancreatic texture (Soft vs. Hard) based on radiomics features.

Main Pancreatic Duct (MPD) Prediction: Estimating the status of the main pancreatic duct using non-invasive imaging data.

✅ Current Status
Univariate analysis shows strong signal in feature association with both outcomes.

Modeling pipeline (train/test split: 70/30) is being developed and tested using the JayaTexture.xlsx dataset.

Focused on the Whipple group for texture classification.

A prospective validation cohort is included for future testing.

🔬 Methods
Radiomics feature extraction from CT scans.

Feature selection based on statistical significance and correlation filtering.

Model development using classification algorithms (e.g., logistic regression, random forest, XGBoost).

Evaluation via AUC, sensitivity, specificity.


![SVM_shap_values_3_features.png](results%2Ftraining_texture%2FSheet1%2FShapley_plots%2FSVM_shap_values_3_features.png)


| **Dataset** | **Classifier** | **\# Features** | **AUC (95% CI)**  | **Sensitivity (95% CI)** | **Specificity (95% CI)** | **PPV (95% CI)**  | **NPV (95% CI)**  |
| ----------- | -------------- | --------------- | ----------------- | ------------------------ | ------------------------ | ----------------- | ----------------- |
| Training    | SVM            | 3               | 0.73 (0.68, 0.77) | 0.80 (0.76, 0.84)        | 0.52 (0.46, 0.57)        | 0.62 (0.57, 0.67) | 0.72 (0.68, 0.77) |
| Test        | SVM            | 3               | 0.69 (0.62, 0.77) | 0.76 (0.69, 0.83)        | 0.48 (0.40, 0.57)        | 0.63 (0.56, 0.71) | 0.63 (0.55, 0.71) |