import numpy as np
import pandas as pd

def calculate_correlation_matrix(df):
    corr_matrix = df.corr().abs()
    return corr_matrix

def select_highly_correlated_features(corr_matrix, threshold=0.9):
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop

# def remove_collinear_features(df, threshold):
#     """
#     Objective:
#         Remove collinear features in a dataframe with a correlation coefficient
#         greater than the threshold.
#     :param x: features dataframe
#     :param threshold: features with correlations greater than this value are removed
#     :return: dataframe that contains only the non-highly-collinear features
#     """
#     case = df.iloc[:, 0]
#     x = df.iloc[:, 1:-1]
#     y = df.iloc[:, -1]
#
#     corr_matrix = x.corr().abs()
#     upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#
#     to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
#     x_dropped = x.drop(columns=to_drop)
#
#     out_df = pd.concat([case, x_dropped, y], axis=1)
#
#     print(f"Features with correlation coefficient larger than {threshold} were removed.")
#     print(f"Among the {x.shape[1]} features, {x_dropped.shape[1]} were selected.")
#
#     return out_df


def remove_collinear_features(df, threshold, priority_features=["Size", "MyGrade"]):
    """
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold, while prioritizing keeping the specified feature.
    :param df: DataFrame containing features and target.
    :param threshold: Threshold for correlation; features with correlations greater than this value are considered collinear.
    :param priority_feature: The feature to prioritize keeping in the dataframe.
    :return: DataFrame containing only non-highly-collinear features.
    """
    case = df.iloc[:, 0]  # Assuming the first column is case identifiers
    x = df.iloc[:, 1:-1]  # Assuming all columns except the first and last are features
    y = df.iloc[:, -1]  # Assuming the last column is the target

    corr_matrix = x.corr().abs()  # Calculate the absolute correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Identify features to drop, prioritizing keeping the priority feature
    to_drop = []
    for column in upper_tri.columns:
        if any(upper_tri[column] > threshold):
            correlated_features = upper_tri.index[upper_tri[column] > threshold].tolist()
            for priority_feature in priority_features:
                if priority_feature in correlated_features:
                    to_drop.append(column)  # Drop the current column if it is correlated with the priority feature
                elif column != priority_feature:
                    to_drop.append(column)  # Otherwise, drop the column if it's not the priority feature

    # Remove duplicates from the drop list
    to_drop = list(set(to_drop))

    # Drop the identified features
    x_dropped = x.drop(columns=to_drop)
    out_df = pd.concat([case, x_dropped, y], axis=1)

    print(f"Features with correlation coefficient larger than {threshold} were removed.")
    print(f"Among the {x.shape[1]} features, {x_dropped.shape[1]} were selected.")

    return out_df



def remove_collinear_features_with_priority(df, threshold, auc_values=None, p_values=None, priority_features=["Size"]):
    case = df.iloc[:, 0]
    x = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    corr_matrix = x.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = []
    for column in upper_tri.columns:
        correlated_features = upper_tri.index[upper_tri[column] > threshold].tolist()

        if correlated_features:
            scores = {}
            for feature in [column] + correlated_features:
                if auc_values is not None and feature in auc_values:
                    scores[feature] = auc_values[feature]
                elif p_values is not None and feature in p_values:
                    scores[feature] = -p_values[feature]  # Negate p-values to treat smaller as better
                elif feature in priority_features:
                    scores[feature] = float('inf')  # Always keep priority features
                else:
                    scores[feature] = 0

            # Keep the best feature and drop the rest
            best_feature = max(scores, key=scores.get)
            to_drop.extend([f for f in [column] + correlated_features if f != best_feature])

    to_drop = list(set(to_drop))
    x_dropped = x.drop(columns=to_drop)
    out_df = pd.concat([case, x_dropped, y], axis=1)

    print(f"Features with correlation coefficient larger than {threshold} were removed.")
    print(f"Among the {x.shape[1]} features, {x_dropped.shape[1]} were selected.")

    return out_df

