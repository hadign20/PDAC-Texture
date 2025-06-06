import os
import pandas as pd
import numpy as np
from jupyterlab.semver import valid

from src.feature_selection.correlation import remove_collinear_features, remove_collinear_features_with_priority
from src.feature_selection.feature_selection import *
from src.model.model_building import evaluate_models, save_classification_results, plot_calibration_curve, \
    ensure_directory_exists
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from src.visualization.plotting import *
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import RFECV

from src.model.model_building import compute_metrics, compute_confidence_interval, remove_outliers, fill_na, normalize
from sklearn.metrics import confusion_matrix




#=========================================
# set paths
#=========================================
data_path = r'D:\projects\PDAC Texture\data'
result_path = r'D:\projects\PDAC Texture\pdacTexture\results'
excel_file_names = ["training_texture"] # training_MDP, training_texture
excel_file_names_val = ["Texture_Test_Isotropic_Sampling_RectangularPrepared"] # Filtered_Texture_Train_UsedForVal, Texture_Test_Isotropic_Sampling_RectangularPrepared

SELECTED_SHEET = [] # "[] for all sheets",  ["3_1"]
SELECTED_SHEET_VAL = [] # "[] for all sheets",  ["3_1"]
outcome_column = "Texture" # Dilated_MPD, Texture
exclude_columns = ["Case"]
exclude_columns_val = ["Case"]
categorical_columns = [] # "Size"
features_to_keep = []


#=========================================
# set parameters
#=========================================
SAVE_PLOTS = True

FEATURE_CORRELATION = True
CORR_THRESH = 0.8

FEATURE_SELECTION = True
FEATURE_SELECTION_METHOD = 'mrmr' # 'mrmr', 'pvalue', 'auc', 'composite'
min_num_features = 3
max_num_features = 3
MRMR_num_features = 15

MODEL_BUILDING = True
RESAMPLING = True
RESAMPLING_METHOD = "RandomOverSampler" # "RandomOverSampler" or "SMOTEENN"
EVALUATION_METHOD = 'train_test_split' # 'train_test_split' or 'cross_validation' or 'cv_feature_selection_model_building'
TEST_SIZE = 0.3
CV_FOLDS = 5
HYPERPARAMETER_TUNING = True

EXTERNAL_VALIDATION = False
RESAMPLING_VAL = False
NORMALIZE_VAL = True
REMOVE_OUTLIER_VAL = False

#=========================================
def save_excel_sheet(df, filepath, sheetname, index=False):
    # Create file if it does not exist
    if not os.path.exists(filepath):
        df.to_excel(filepath, sheet_name=sheetname, index=index)

    # Otherwise, add a sheet. Overwrite if there exists one with the same name.
    else:
        with pd.ExcelWriter(filepath, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer:
            df.to_excel(writer, sheet_name=sheetname, index=index)

def save_summary_results(classification_results, evaluation_method, results_dir, summary_results, sheet, num_features):

    if evaluation_method == "cross_validation":
        for classifier, result in classification_results.items():
            result_entry = {
                'Sheet': sheet,
                'Num Features': num_features,
                'Classifier': classifier,
                'AUC': result['metrics']['roc_auc'],
                'Sensitivity': result['metrics']['sensitivity'],
                'Specificity': result['metrics']['specificity'],
                'PPV': result['metrics']['ppv'],
                'NPV': result['metrics']['npv']
            }
            summary_results.append(result_entry)

    elif evaluation_method == "train_test_split":
        for dataset in classification_results.items():
            if dataset[0] == "test":
                for classifier, result in dataset[1].items():
                    result_entry = {
                        'Sheet': sheet,
                        'Num Features': num_features,
                        'Classifier': classifier,
                        'AUC': result['metrics']['roc_auc'],
                        'Sensitivity': result['metrics']['sensitivity'],
                        'Specificity': result['metrics']['specificity'],
                        'PPV': result['metrics']['ppv'],
                        'NPV': result['metrics']['npv']
                    }
                    summary_results.append(result_entry)

    # Save summary results
    summary_df = pd.DataFrame(summary_results)
    summary_file = os.path.join(results_dir, 'summary_results.xlsx')
    with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
        for sheet_name in summary_df['Sheet'].unique():
            sheet_df = summary_df[summary_df['Sheet'] == sheet_name]
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Sort all results by AUC and save to "Best Result" sheet
        best_df = summary_df.sort_values(by='AUC', ascending=False)
        best_df.to_excel(writer, sheet_name='Best Result', index=False)

def remove_outliers(df):
    out_df = df.copy()

    # IQR
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        out_df[col] = np.where(out_df[col] < lower, lower, out_df[col])
        out_df[col] = np.where(out_df[col] > upper, upper, out_df[col])

    return out_df

def normalize(X: pd.DataFrame):
    #scaler = MinMaxScaler()
    scaler = StandardScaler()
    cols = X.columns
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=cols)
    return X

def normalize_df(df: pd.DataFrame, outcome_column: str, exclude_columns: list):
    excluded_columns = exclude_columns + [outcome_column]
    cols_to_scale = [col for col in df.columns if col not in excluded_columns]
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[cols_to_scale])
    scaled_df = pd.DataFrame(scaled_values, columns=cols_to_scale, index=df.index)
    result_df = pd.concat([df[exclude_columns], scaled_df, df[[outcome_column]]], axis=1)
    return result_df


#=========================================


def main():
    for excel_file_name in excel_file_names:
        print("\n======================================================================")
        features_file = os.path.join(data_path, excel_file_name + ".xlsx")
        results_dir = os.path.join(result_path, excel_file_name)
        os.makedirs(results_dir, exist_ok=True)
        xls = pd.ExcelFile(features_file)
        summary_results = []
        summary_results_val = []

        selected_sheets = xls.sheet_names if len(SELECTED_SHEET) == 0 else SELECTED_SHEET

        for sheet in selected_sheets:
            result_dir = os.path.join(results_dir, sheet)
            os.makedirs(result_dir, exist_ok=True)

            df = pd.read_excel(xls, sheet_name=sheet)
            df = df.fillna(0)

            # =========================================
            # Feature selection
            # =========================================
            if FEATURE_CORRELATION:
                df = normalize_df(df, exclude_columns=exclude_columns, outcome_column=outcome_column)

                # select significant features first
                p_values_df = calculate_p_values(df, outcome_column, categorical_columns, exclude_columns)
                p_values_df = p_values_df[p_values_df['P_Value'] <= 0.05]
                selected_features = p_values_df['Feature'].tolist()
                for f in features_to_keep:
                    if f not in selected_features:
                        print(f"feature {f} was not selected in p-value analysis")
                        selected_features.append(f)
                df = df[exclude_columns + selected_features + [outcome_column]]

                print("\n======================================================================")
                print(f"Removing correlated features for sheet {sheet}")
                print("======================================================================")

                df_before_correlation_removal = df.copy()
                df = remove_collinear_features(df, CORR_THRESH)
                for f in features_to_keep:
                    if f not in df.columns:
                        print(f"feature {f} was removed due to high correlation...")
                        removed_col = df_before_correlation_removal[f]
                        df = pd.concat([df, removed_col], axis=1)
                df[outcome_column] = df.pop(outcome_column)



                # # ------------------------------
                # print(f"Removing correlated features for file {excel_file_name}, sheet {sheet}")
                # df_before_correlation_removal = df.copy()
                # priority_features = ["Size"]
                # p_values_df = calculate_p_values(df, outcome_column, categorical_columns, exclude_columns)
                # auc_values_df = calculate_auc_values(df, outcome_column, categorical_columns, exclude_columns)
                #
                # p_values_dict = dict(zip(p_values_df['Feature'], p_values_df['P_Value']))
                # auc_values_dict = dict(zip(auc_values_df['Feature'], auc_values_df['AUC']))
                #
                # df = remove_collinear_features_with_priority(
                #     df=df,
                #     threshold=CORR_THRESH,
                #     auc_values=auc_values_dict,
                #     p_values=p_values_dict,
                #     priority_features=priority_features
                # )
                #
                # for f in features_to_keep:
                #     if f not in df.columns:
                #         print(f"feature {f} was removed due to high correlation...")
                #         removed_col = df_before_correlation_removal[f]
                #         df = pd.concat([df, removed_col], axis=1)
                # df[outcome_column] = df.pop(outcome_column)




            if FEATURE_SELECTION:
                print(f"Performing feature analysis for file {excel_file_name}, sheet {sheet}")
                p_values_df = calculate_p_values(df, outcome_column, categorical_columns, exclude_columns)
                auc_values_df = calculate_auc_values_CV(df, outcome_column, categorical_columns, exclude_columns)
                mrmr_df = MRMR_feature_count(df, outcome_column, categorical_columns, exclude_columns,
                                             MRMR_num_features, CV_FOLDS)
                # lasso_df = lasso_feature_selection(df, outcome_column, exclude_columns)
                # composite_df = calculate_feature_scores(p_values_df, auc_values_df, mrmr_df, lasso_df, result_dir)
                composite_df = calculate_feature_scores(p_values_df, auc_values_df, mrmr_df, result_dir)

                output_file = os.path.join(results_dir, f'mrmr_features.xlsx')
                save_excel_sheet(mrmr_df, output_file, sheetname=sheet)

                save_feature_analysis(p_values_df, auc_values_df, mrmr_df, composite_df, result_dir)

                df_copy = df.copy()

                for feature_num in range(min_num_features, max_num_features + 1):
                    print(f"Selecting {feature_num} significant features for file {excel_file_name}, sheet {sheet}")

                    num_features = feature_num

                    selected_features = []
                    if FEATURE_SELECTION_METHOD == 'mrmr':
                        selected_features = mrmr_df['Feature'][:num_features].tolist()
                        print(f"{num_features} features were selected by using MRMR method")
                    elif FEATURE_SELECTION_METHOD == 'pvalue':
                        selected_features = p_values_df['Feature'][:num_features].tolist()
                        print(f"{num_features} features were selected by using pvalue method")
                    elif FEATURE_SELECTION_METHOD == 'auc':
                        selected_features = auc_values_df['Feature'][:num_features].tolist()
                        print(f"{num_features} features were selected by using auc method")
                    # elif FEATURE_SELECTION_METHOD == 'lasso':
                    #     selected_features = lasso_df['Feature'][:num_features].tolist()
                    #     print(f"{num_features} features were selected by using lasso method")
                    elif FEATURE_SELECTION_METHOD == 'composite':
                        selected_features = composite_df['Feature'][:num_features].tolist()
                        print(f"{num_features} features were selected by a composite of p_value, AUC, and MRMR method")
                    else:
                        raise ValueError(
                            "FEATURE_SELECTION_METHOD is not correct. It should be 'mrmr', 'pvalue', 'auc', or 'composite'")

                    for f in features_to_keep:
                        if f not in selected_features:
                            print(f"feature {f} was not selected in feature selection")
                            selected_features.append(f)
                            num_features += 1

                    df = df_copy[exclude_columns + selected_features + [outcome_column]]

                    # =========================================
                    # Model building and evaluation
                    # =========================================
                    if MODEL_BUILDING:
                        resampling_method_ra = None
                        if RESAMPLING:
                            if RESAMPLING_METHOD == "RandomOverSampler":
                                resampling_method_ra = RandomOverSampler(random_state=42)
                            elif RESAMPLING_METHOD == "SMOTEENN":
                                resampling_method_ra = SMOTEENN(random_state=42)

                        eval_kwargs = None
                        if EVALUATION_METHOD == 'train_test_split':
                            eval_kwargs = {'test_size': TEST_SIZE,
                                           'random_state': 42,
                                           'result_path': result_dir,
                                           'num_features': num_features,
                                           'tuning': HYPERPARAMETER_TUNING,
                                           'resampling_method': resampling_method_ra}
                        elif EVALUATION_METHOD == 'cross_validation':
                            eval_kwargs = {'cv_folds': CV_FOLDS,
                                            'result_path': result_dir,
                                            'num_features': num_features,
                                            'tuning': HYPERPARAMETER_TUNING,
                                            'resampling_method': resampling_method_ra}
                        else:
                            eval_kwargs = {'cv_folds': CV_FOLDS,
                                           'result_path': result_dir,
                                           'num_features': num_features,
                                           'tuning': HYPERPARAMETER_TUNING,
                                           'resampling_method': resampling_method_ra}



                        print("\n======================================================================")
                        print(f"Training and evaluating classification models for {num_features} feature(s) in sheet {sheet}")
                        print("======================================================================")
                        if EVALUATION_METHOD == 'cv_feature_selection_model_building':
                            df = df_copy
                        # X = df.loc[:, ~df.columns.isin(exclude_columns + [outcome_column])]
                        X = df.loc[:, ~df.columns.isin([outcome_column])]
                        y = df[outcome_column]

                        classification_results = evaluate_models(X, y, method=EVALUATION_METHOD, **eval_kwargs)

                        classification_results_file = os.path.join(result_dir, 'model_evaluation_results.xlsx')
                        save_classification_results(classification_results, classification_results_file, num_features, method=EVALUATION_METHOD)


                        if EVALUATION_METHOD == "cross_validation" or EVALUATION_METHOD == 'cv_feature_selection_model_building':
                            for classifier, result in classification_results.items():
                                result_entry = {
                                    'Sheet': sheet,
                                    'Num Features': num_features,
                                    'Classifier': classifier,
                                    'AUC': result['metrics']['roc_auc'],
                                    'Sensitivity': result['metrics']['sensitivity'],
                                    'Specificity': result['metrics']['specificity'],
                                    'PPV': result['metrics']['ppv'],
                                    'NPV': result['metrics']['npv'],
                                    'F-Score': result['metrics']['f1_score']
                                }
                                summary_results.append(result_entry)
                        elif EVALUATION_METHOD == "train_test_split":
                            for dataset in classification_results.items():
                                dataset_name = dataset[0]
                                for classifier, result in dataset[1].items():
                                    result_entry = {
                                        'Sheet': f"{sheet}_{dataset_name}",
                                        'Dataset': dataset_name.capitalize(),
                                        'Num Features': num_features,
                                        'Classifier': classifier,
                                        'AUC': result['metrics']['roc_auc'],
                                        'Sensitivity': result['metrics']['sensitivity'],
                                        'Specificity': result['metrics']['specificity'],
                                        'PPV': result['metrics']['ppv'],
                                        'NPV': result['metrics']['npv'],
                                    }
                                    summary_results.append(result_entry)

                        # Save summary results
                        summary_df = pd.DataFrame(summary_results)
                        summary_file = os.path.join(results_dir, 'summary_results.xlsx')

                        with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
                            for dataset_type in ['train', 'test']:
                                dataset_df = summary_df[summary_df['Dataset'] == dataset_type.capitalize()]
                                dataset_df = dataset_df.sort_values(by='AUC', ascending=False)
                                dataset_df.to_excel(writer, sheet_name=f"{dataset_type.capitalize()} Results",
                                                    index=False)

                            # # Best results (across all datasets)
                            # best_df = summary_df.sort_values(by='AUC', ascending=False)
                            # best_df.to_excel(writer, sheet_name='Best Result', index=False)

                    # =========================================
                    # External Validation Cohort
                    # =========================================
                    if EXTERNAL_VALIDATION:
                        roc_data_path_val = os.path.join(result_path, excel_file_name, sheet, "ROC_data_val")
                        if not os.path.exists(roc_data_path_val): os.makedirs(roc_data_path_val)

                        prob_data_path_val = os.path.join(result_path, excel_file_name, sheet, "Prob_data_val")
                        if not os.path.exists(prob_data_path_val): os.makedirs(prob_data_path_val)

                        for excel_file_name_val in excel_file_names_val:
                            features_file_val = os.path.join(data_path, excel_file_name_val + ".xlsx")
                            xls_val = pd.ExcelFile(features_file_val)
                            selected_sheets_val = xls_val.sheet_names if len(SELECTED_SHEET_VAL) == 0 else SELECTED_SHEET_VAL

                            out_dict = {}
                            for sh in selected_sheets_val:
                                #if sh != sheet: continue
                                rows = []
                                validation_df = pd.read_excel(features_file_val, sheet_name=sh)
                                validation_df = validation_df.fillna(0)

                                #models = ['RandomForest', 'SVM', 'LogisticRegression', 'NaiveBayes', 'MLP']
                                #models = ['RandomForest', 'SVM', 'LogisticRegression', 'NaiveBayes']
                                models = ['LogisticRegression']


                                for model_name in models:
                                    model_file = os.path.join(result_path, excel_file_name, sheet, "Saved_Models",
                                                              model_name + '_' + str(num_features) + '_features.pkl')

                                    print(f"Testing trained {model_name} on {excel_file_name_val}, sheet {sh} with {num_features} features...")

                                    model = joblib.load(model_file)
                                    f_names = model.feature_names
                                    train_threshold = model.selected_thresh


                                    X_validation = validation_df[f_names]
                                    y_validation = validation_df[outcome_column]

                                    if REMOVE_OUTLIER_VAL: X_validation = remove_outliers(X_validation)
                                    if NORMALIZE_VAL:
                                        X_validation = normalize(X_validation)
                                        # X_validation = normalize_df(X_validation, exclude_columns=exclude_columns,
                                        #                   outcome_column=outcome_column)
                                    if RESAMPLING_VAL:
                                        resampling_method = RandomOverSampler(random_state=42)
                                        X_validation, y_validation = resampling_method.fit_resample(X_validation, y_validation)




                                    #--------------------------------------
                                    predicted_probs = model.predict_proba(X_validation[f_names])[:, 1]

                                    # Compute TPR and FPR for ROC curve
                                    fpr, tpr, thresholds = roc_curve(y_validation, predicted_probs)

                                    # Compute AUC for ROC curve
                                    roc_auc = auc(fpr, tpr)

                                    # Confidence interval calculation with stratified bootstrap resampling
                                    bootstrapped_tpr = []
                                    n_bootstraps = 1000  # Increase the number of bootstraps for more stability
                                    rng_seed = 42  # Random seed for reproducibility
                                    rng = np.random.RandomState(rng_seed)

                                    for i in range(n_bootstraps):
                                        # Stratified bootstrapping to maintain class distribution
                                        indices = rng.choice(np.arange(len(y_validation)), size=len(y_validation),
                                                             replace=True)
                                        y_sample = y_validation.iloc[indices]
                                        if len(np.unique(y_sample)) < 2:
                                            # We need at least one positive and one negative sample to compute ROC
                                            continue
                                        fpr_boot, tpr_boot, _ = roc_curve(y_sample, predicted_probs[indices])
                                        bootstrapped_tpr.append(
                                            np.interp(fpr, fpr_boot, tpr_boot))  # Interpolation to align FPR

                                    bootstrapped_tpr = np.array(bootstrapped_tpr)
                                    tpr_lower = np.percentile(bootstrapped_tpr, 2.5, axis=0)
                                    tpr_upper = np.percentile(bootstrapped_tpr, 97.5, axis=0)

                                    # Create a DataFrame with TPR, FPR, AUC, and confidence intervals
                                    roc_df = pd.DataFrame({
                                        'fpr': fpr,
                                        'tpr': tpr,
                                        'thresholds': thresholds,
                                        'tpr_lower': tpr_lower,
                                        'tpr_upper': tpr_upper,
                                    })

                                    # Add the AUC value to a separate column (you can also store it separately if needed)
                                    roc_df['AUC'] = roc_auc

                                    # Define the ROC file path and save it
                                    roc_file = os.path.join(roc_data_path_val,
                                                            f'ROC_{model_name}_{sh}_{num_features}.xlsx')
                                    roc_df.to_excel(roc_file, index=False)
                                    # --------------------------------------

                                    metrics, ci, y_pred_val_optimal = compute_metrics(y_validation, predicted_probs, predefined_thresh=train_threshold, target_sensitivity=0.8)
                                    #metrics, ci, y_pred_val_optimal = compute_metrics(y_validation, predicted_probs, target_sensitivity=0.8)

                                    ### Save case_id, predicted outcome, actual outcome, and probability scores
                                    # output_df = pd.DataFrame({
                                    #     'Case_ID': validation_df['Case_ID'],  # Replace 'Case_ID' if column name differs
                                    #     'Probability_Score': predicted_probs,
                                    #     'Predicted_Outcome': y_pred_val_optimal,  # Predicted outcome
                                    #     'Actual_Outcome': y_validation  # Actual outcome
                                    # })



                                    output_df = pd.DataFrame({
                                        'Case_ID': validation_df['Case_ID'],  # Replace 'Case_ID' if column name differs
                                        'Probability_Score': model.predict_proba(X_validation)[:, 1],
                                        'Predicted_Outcome': model.predict(X_validation),  # Predicted outcome
                                        'Actual_Outcome': y_validation  # Actual outcome
                                    })





                                    output_excel_path_case = os.path.join(prob_data_path_val,
                                                                          f"{sh}_case_predictions_{model_name}_{num_features}.xlsx")
                                    save_excel_sheet(output_df, output_excel_path_case, str(num_features), False)


                                    # cm_matrix = pd.DataFrame({
                                    #     'TP': [f"{metrics.get('tp', 'N/A')}"],
                                    #     'TN': [f"{metrics.get('tn', 'N/A')}"],
                                    #     'FP': [f"{metrics.get('fp', 'N/A')}"],
                                    #     'FN': [f"{metrics.get('fn', 'N/A')}"],
                                    #     'Threshold': [f"{metrics.get('target_threshold', 'N/A'):.8f}"],
                                    # })

                                    cm = confusion_matrix(y_validation, model.predict(X_validation))
                                    tn, fp, fn, tp = cm.ravel()
                                    cm_matrix = pd.DataFrame({
                                        'TP': [f"{tp}"],
                                        'TN': [f"{tn}"],
                                        'FP': [f"{fp}"],
                                        'FN': [f"{fn}"],
                                        'Threshold': [f"{metrics.get('target_threshold', 'N/A'):.8f}"],
                                    })

                                    save_excel_sheet(cm_matrix, output_excel_path_case, "cm_" + str(num_features),False)
                                    # --------------------------------------
                                    if SAVE_PLOTS:
                                        calibration_path = os.path.join(result_dir, "Calibration_plots")
                                        dca_path = os.path.join(result_dir, "DCA_curves")
                                        ensure_directory_exists(dca_path)

                                        #plot_calibration_curve(y_validation, predicted_probs, model_name, num_features, output_dir=calibration_path)
                                        plot_dca(y_validation, predicted_probs, model_name, num_features, sh, output_dir=dca_path)



                                        umap_path = os.path.join(result_dir, "UMAP_plots", "Validation")
                                        ensure_directory_exists(umap_path)


                                        val_df = pd.concat([X_validation, y_validation], axis = 1)
                                        output_validation_df = os.path.join(umap_path,
                                                                              f"{sh}_validationDF_{num_features}.xlsx")
                                        save_excel_sheet(val_df, output_validation_df, sh, False)

                                        # plot_umap(val_df, model_name, num_features, sh, features=f_names, exclude_cols=exclude_columns + features_to_keep, outcome_col=outcome_column,
                                        #           title='UMAP Projection of Radiomics Features for Validation Cohort', filepath=umap_path)
                                        #
                                        # plot_umap_tsne_3d(val_df, model_name, num_features, sh, features=f_names,
                                        #                   exclude_cols=exclude_columns + features_to_keep,
                                        #                   outcome_col=outcome_column,
                                        #                   title='UMAP Projection of Radiomics Features for Validation Cohort',
                                        #                   filepath=umap_path)



                                        waterfall_path = os.path.join(result_dir, "Waterfall_plots", "Validation")
                                        ensure_directory_exists(waterfall_path)
                                        plot_waterfall(y_validation, predicted_probs, model_name, num_features, sheet=sh,
                                                       pos_label="PNET grade high", neg_label="PNET grade low",
                                                       output_dir=waterfall_path,
                                                       threshold=metrics.get('target_threshold', 'N/A'))

                                    # --------------------------------------





                                    row = [
                                        model_name,
                                        f"{metrics.get('roc_auc', 'N/A'):.2f} ({ci.get('roc_auc', ['N/A', 'N/A'])[0]:.2f}, {ci.get('roc_auc', ['N/A', 'N/A'])[1]:.2f})",
                                        f"{metrics.get('sensitivity', 'N/A'):.2f} ({ci.get('sensitivity', ['N/A', 'N/A'])[0]:.2f}, {ci.get('sensitivity', ['N/A', 'N/A'])[1]:.2f})",
                                        f"{metrics.get('specificity', 'N/A'):.2f} ({ci.get('specificity', ['N/A', 'N/A'])[0]:.2f}, {ci.get('specificity', ['N/A', 'N/A'])[1]:.2f})",
                                        f"{metrics.get('ppv', 'N/A'):.2f} ({ci.get('ppv', ['N/A', 'N/A'])[0]:.2f}, {ci.get('ppv', ['N/A', 'N/A'])[1]:.2f})",
                                        f"{metrics.get('npv', 'N/A'):.2f} ({ci.get('npv', ['N/A', 'N/A'])[0]:.2f}, {ci.get('npv', ['N/A', 'N/A'])[1]:.2f})",
                                        f"{metrics.get('f1_score', 'N/A'):.2f} ({ci.get('f1_score', ['N/A', 'N/A'])[0]:.2f}, {ci.get('f1_score', ['N/A', 'N/A'])[1]:.2f})",
                                        f"{metrics.get('tp', 'N/A')}",
                                        f"{metrics.get('tn', 'N/A')}",
                                        f"{metrics.get('fp', 'N/A')}",
                                        f"{metrics.get('fn', 'N/A')}",
                                        f"{metrics.get('target_threshold', 'N/A'):.2f}"
                                    ]
                                    rows.append(row)

                                    result_entry_val = {
                                        'Sheet': sheet,
                                        'Val_Sheet': sh,
                                        'Num Features': num_features,
                                        'Classifier': model_name,
                                        'AUC': metrics.get('roc_auc', 'N/A'),
                                        'Sensitivity': metrics.get('sensitivity', 'N/A'),
                                        'Specificity': metrics.get('specificity', 'N/A'),
                                        'PPV': metrics.get('ppv', 'N/A'),
                                        'NPV': metrics.get('npv', 'N/A'),
                                        'F-Score': metrics.get('f1_score', 'N/A')
                                    }
                                    summary_results_val.append(result_entry_val)

                                    # Save summary results
                                    summary_df_val = pd.DataFrame(summary_results_val)
                                    summary_file_val = os.path.join(results_dir, 'summary_results_val.xlsx')
                                    with pd.ExcelWriter(summary_file_val, engine='openpyxl') as writer:
                                        #best_df_val = summary_df_val.sort_values(by='F-Score', ascending=False)
                                        best_df_val = summary_df_val.sort_values(by='AUC', ascending=False)
                                        best_df_val.to_excel(writer, sheet_name='Best Result Val', index=False)

                                df = pd.DataFrame(rows, columns=['Classifier', 'AUC (95% CI)', 'Sensitivity (95% CI)',
                                                                 'Specificity (95% CI)',
                                                                 'PPV (95% CI)', 'NPV (95% CI)', 'F-Score', 'TP', 'TN', 'FP', 'FN', 'Optimal_Thresh'])

                                output_excel_path = os.path.join(result_path, excel_file_name, sheet,
                                                                 f"{sh}_external_validation_results.xlsx")
                                save_excel_sheet(df, output_excel_path, str(num_features) + "_features", False)


if __name__ == '__main__':
    main()

