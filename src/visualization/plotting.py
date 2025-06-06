import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import os
import pandas as pd
from dcurves import dca, plot_graphs, load_test_data
import umap


def plot_boxplot(data, x, y, hue=None, title=None, figsize=(10, 6), save_path=None):
    """Plot and save a boxplot."""
    plt.figure(figsize=figsize)
    sns.boxplot(data=data, x=x, y=y, hue=hue)
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_facet_grid_boxplot(data, x, y, hue=None, col=None, col_wrap=4, order=None, hue_order=None, title=None, save_path=None):
    """Plot and save a FacetGrid of boxplots."""
    g = sns.FacetGrid(data, col=col, col_wrap=col_wrap, height=4, aspect=1.2)
    g.map(sns.boxplot, x, y, hue, order=order, hue_order=hue_order)
    g.add_legend()
    g.set_titles("{col_name}")
    if title:
        g.fig.suptitle(title, y=1.05)
    if save_path:
        g.savefig(save_path)
    plt.close()


def plot_auc_with_ci(y_true, y_pred, n_bootstraps=1000, alpha=0.95):
    bootstrapped_scores = []
    rng = np.random.RandomState(42)

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue

        score = auc(*roc_curve(y_true[indices], y_pred[indices])[:2])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    lower_bound = sorted_scores[int((1.0 - alpha) / 2 * len(sorted_scores))]
    upper_bound = sorted_scores[int((1.0 + alpha) / 2 * len(sorted_scores))]

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.fill_between(fpr, tpr, alpha=0.2, label=f'{alpha * 100:.1f}% CI [{lower_bound:0.2f} - {upper_bound:0.2f}]')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_kaplan_meier(time, status, save_path=None):
    """
    Plot Kaplan-Meier survival curve.

    Parameters:
    time (array-like): Survival time data
    status (array-like): Event occurrence data (1 if event occurred, 0 if censored)
    """
    # Sort the data by time
    sorted_indices = np.argsort(time)
    time = np.array(time)[sorted_indices]
    status = np.array(status)[sorted_indices]

    # Number of subjects at time 0
    n = len(time)

    # Initialize the survival probability and times
    survival_prob = np.ones(n + 1)
    survival_times = np.concatenate(([0], time))

    # Calculate survival probability at each time point
    for i in range(1, n + 1):
        survival_prob[i] = survival_prob[i - 1] * (1 - status[i - 1] / n)
        n -= 1

    # Plot the Kaplan-Meier curve
    plt.step(survival_times, survival_prob, where="post")
    plt.title("Kaplan-Meier Curve")
    plt.xlabel("Time (Months)")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=600)
    #plt.show()




def plot_waterfall1(y_true, y_pred_prob, classifier_name, num_features, output_dir='./plots'):
    """
    Generate and save a waterfall plot.

    Parameters:
        y_true (array-like): True labels (binary: 0 or 1).
        y_pred_prob (array-like): Predicted probabilities.
        classifier_name (str): Name of the classifier.
        num_features (int): Number of features used in the model.
        output_dir (str): Directory where the plot will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create DataFrame and sort values
    df = pd.DataFrame({'y_true': y_true, 'y_pred_prob': y_pred_prob})
    df = df.sort_values(by='y_pred_prob', ascending=False).reset_index(drop=True)

    # Compute the contribution to the waterfall plot
    df['contribution'] = df['y_pred_prob'].diff().fillna(df['y_pred_prob'].iloc[0])

    # Create figure
    plt.figure(figsize=(10, 6))

    # Define waterfall bars
    bars = range(len(df))
    plt.bar(bars, df['contribution'], color=['green' if y else 'red' for y in df['y_true']])

    # Line showing cumulative sum
    plt.plot(bars, df['y_pred_prob'], marker='o', linestyle='dashed', color='blue', label='Cumulative Probability')

    # Labels and title
    plt.xlabel('Sorted Predictions')
    plt.ylabel('Predicted Probability Contribution')
    plt.title(f'Waterfall Plot - {classifier_name} ({num_features} Features)')
    plt.legend()

    # Save figure
    filepath = os.path.join(output_dir, f'waterfall_{classifier_name}_{num_features}_features.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Waterfall plot saved to: {filepath}")




import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_waterfall2(y_true, y_pred_prob, classifier_name, num_features, pos_label, neg_label, sheet="", output_dir='./plots', threshold=None):
    """
    Generate and save a waterfall plot.

    Parameters:
        y_true (array-like): True labels (binary: 0 or 1).
        y_pred_prob (array-like): Predicted radiomic scores.
        classifier_name (str): Name of the classifier.
        num_features (int): Number of features used in the model.
        output_dir (str): Directory where the plot will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create DataFrame and sort values
    df = pd.DataFrame({'y_true': y_true, 'y_pred_prob': y_pred_prob})
    df = df.sort_values(by='y_pred_prob', ascending=False).reset_index(drop=True)

    # Define colors based on y_true
    colors = ['red' if label == 1 else 'blue' for label in df['y_true']]

    # Create figure
    plt.figure(figsize=(12, 7))
    plt.bar(range(len(df)), df['y_pred_prob'], color=colors)

    # Labels and title with larger font sizes
    plt.xlabel('Patient Number', fontsize=32)
    plt.ylabel('Radiomic Score', fontsize=32)
    plt.title(f'Waterfall Plot - {classifier_name} ({num_features} Features)', fontsize=36)
    plt.ylim(0, 1)

    # Ticks font size
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # Legend
    legend_elements = [Patch(facecolor='red', label=pos_label),
                       Patch(facecolor='blue', label=neg_label)]
    plt.legend(handles=legend_elements, loc='best', fontsize=24)

    # Save figure
    filepath = os.path.join(output_dir, f'waterfall2_{classifier_name}_{sheet}_{num_features}_features.png')
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"Waterfall plot saved to: {filepath}")






def plot_waterfall(y_true, y_pred_prob, classifier_name, num_features,  pos_label, neg_label, sheet="", output_dir='./plots',
                   threshold=None):
    """
    Generate and save a waterfall plot.

    Parameters:
        y_true (array-like): True labels (binary: 0 or 1).
        y_pred_prob (array-like): Predicted radiomic scores.
        classifier_name (str): Name of the classifier.
        num_features (int): Number of features used in the model.
        pos_label (str): Label for positive class (legend).
        neg_label (str): Label for negative class (legend).
        output_dir (str): Directory where the plot will be saved.
        threshold (float, optional): Threshold value for classification.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create DataFrame and sort by predicted score
    df = pd.DataFrame({'y_true': y_true, 'y_pred_prob': y_pred_prob})
    df = df.sort_values(by='y_pred_prob', ascending=True).reset_index(drop=True)

    # Compute difference from threshold
    if threshold is not None:
        df['diff'] = df['y_pred_prob'] - threshold
    else:
        df['diff'] = df['y_pred_prob']

    # Define colors based on y_true
    colors = ['blue' if label == 0 else 'orange' for label in df['y_true']]

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot bars starting from the threshold
    plt.bar(range(len(df)), df['diff'], color=colors, bottom=threshold)

    # Plot threshold line
    if threshold is not None:
        plt.axhline(y=threshold, color='red', linestyle='dashed', label=f'Threshold = {threshold}')

    # Labels and title
    ax = plt.gca()
    ax.set_xlabel('Patients', fontsize=24, labelpad=20)
    ax.set_ylabel('Probability Scores', fontsize=24, labelpad=20)

    plt.title(f'Waterfall Chart - {classifier_name} (Validation Cohort)', fontsize=28, pad=20)
    plt.ylim(0, 1)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label=neg_label),
                       Patch(facecolor='orange', label=pos_label)]
    plt.legend(handles=legend_elements, loc='best', fontsize=18)

    # Save figure
    filepath = os.path.join(output_dir, f'waterfall1_{classifier_name}_{sheet}_{num_features}_features.png')
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"Waterfall plot saved to: {filepath}")





def plot_dca(y_true, y_pred_prob, classifier_name, num_features, sheet, output_dir='./plots'):
    thresholds = np.linspace(0.01, 0.99, 999)
    thresholds = np.arange(0, 0.99, 0.01)

    df_binary = pd.DataFrame(columns=['outcome', 'model'])
    df_binary['outcome'] = y_true
    df_binary['model'] = y_pred_prob

    df_dca = \
        dca(
            data=df_binary,
            outcome='outcome',
            modelnames=['model'],
            thresholds=thresholds,
        )

    # Plot DCA curve
    plt.figure(figsize=(8, 6))
    plot_graphs(
        plot_df=df_dca,
        graph_type='net_benefit',
        y_limits=[-0.05, 0.5]
    )

    # Save figure
    filepath = os.path.join(output_dir, f'dca_{classifier_name}_{sheet}_{num_features}_features.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_dca1(y_true, y_pred_prob, classifier_name, num_features, sheet, output_dir='./plots'):
    filepath = os.path.join(output_dir, f'dca_{classifier_name}_{num_features}_features.png')

    thresholds = np.linspace(0.01, 0.99, 99)
    net_benefits = []
    treat_all = []

    # Calculate prevalence (proportion of positive cases)
    prevalence = np.mean(y_true)

    for threshold in thresholds:
        tp = ((y_pred_prob >= threshold) & (y_true == 1)).sum()
        fp = ((y_pred_prob >= threshold) & (y_true == 0)).sum()

        # Net benefit calculation for the model
        if len(y_true) > 0:
            net_benefit = (tp / len(y_true)) - (fp / len(y_true)) * (threshold / (1 - threshold))
        else:
            net_benefit = 0

        # "Treat all" strategy: everyone is classified as positive
        # Net benefit for treat all = prevalence - (1-prevalence) * threshold/(1-threshold)
        treat_all_net_benefit = prevalence - (1 - prevalence) * (threshold / (1 - threshold))

        treat_all.append(treat_all_net_benefit)
        net_benefits.append(net_benefit)

    # Plot DCA curve
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, net_benefits, label='Model', color='blue', linewidth=2)
    plt.fill_between(thresholds, 0, net_benefits, color='red', alpha=0.2)
    plt.plot(thresholds, treat_all, label='Treat all', color='black', linestyle='-', linewidth=2)
    plt.axhline(y=0, color='gray', linestyle='--', label='Treat none', linewidth=2)

    plt.xlabel('Threshold Probability', fontsize=14)
    plt.ylabel('Net Benefit', fontsize=14)
    plt.title(f'Model {classifier_name} DCA', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(filepath)
    plt.close()






def plot_umap1(df, classifier_name, num_features, sheet="", features=None, exclude_cols=None, outcome_col='', title='UMAP Projection of Radiomics Features', filepath='./plots'):
    """
    Generates a UMAP projection for selected radiomics features.

    Parameters:
    - df (pd.DataFrame): Dataset containing radiomics features and outcome variable.
    - classifier_name (str): Name of the classifier (for file naming).
    - num_features (int): Number of features used.
    - sheet (str): Sheet name or identifier (for file naming).
    - features (list, optional): List of selected features to use for UMAP.
    - exclude_cols (list, optional): List of columns to exclude.
    - outcome_col (str): Column name used for color mapping.
    - title (str): Title of the UMAP plot.
    - filepath (str): Path to save the plot.

    Returns:
    - None (saves the UMAP plot to file).
    """

    # Ensure exclude_cols is not None
    if exclude_cols is None:
        exclude_cols = []

    # Identify feature columns, excluding outcome and any specified columns
    if features is None:
        features = [col for col in df.columns if col not in exclude_cols + [outcome_col]]

    # Ensure features exist in the dataset
    features = [col for col in features if col in df.columns]

    # Print debugging info
    print(f"Selected features ({len(features)}): {features}")

    # Extract features and ensure they are numeric
    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)  # Convert to numeric, handle NaNs

    # Check if X is empty (debugging step)
    if X.empty or X.shape[1] == 0:
        raise ValueError("No valid numeric features available for UMAP.")

    # Extract outcome column
    y = df[outcome_col] if outcome_col in df.columns else None

    # Apply UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

    embedding = reducer.fit_transform(X)

    # Create a DataFrame for plotting
    umap_df = pd.DataFrame(embedding, columns=['UMAP-1', 'UMAP-2'])

    if y is not None:
        umap_df[outcome_col] = y.values  # Add outcome variable for coloring

    # Plot UMAP
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='UMAP-1', y='UMAP-2',
        hue=umap_df[outcome_col] if outcome_col in umap_df else None,
        palette='coolwarm',
        data=umap_df,
        alpha=0.8
    )

    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title=outcome_col if outcome_col in umap_df else None)

    # Ensure filepath exists before saving
    os.makedirs(filepath, exist_ok=True)
    save_path = os.path.join(filepath, f'umap_{classifier_name}_{sheet}_{num_features}_features.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"UMAP plot saved to {save_path}")





def plot_umap(df, classifier_name, num_features, sheet="", features=None, exclude_cols=None, outcome_col='',
              title='UMAP Projection of Radiomics Features', filepath='./plots'):
    """
    Generates a high-quality UMAP plot for selected radiomics features.
    """

    # Ensure exclude_cols is not None
    if exclude_cols is None:
        exclude_cols = []

    # Identify feature columns
    if features is None:
        features = [col for col in df.columns if col not in exclude_cols + [outcome_col]]
    features = [col for col in features if col in df.columns]

    print(f"Selected features ({len(features)}): {features}")

    # Convert to numeric and handle missing values
    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)

    if X.empty or X.shape[1] == 0:
        raise ValueError("No valid numeric features available for UMAP.")

    # Extract outcome column
    y = df[outcome_col] if outcome_col in df.columns else None

    # UMAP embedding
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(X)

    # Prepare DataFrame for plotting
    umap_df = pd.DataFrame(embedding, columns=['UMAP-1', 'UMAP-2'])
    if y is not None:
        umap_df[outcome_col] = y.values

    # Set up plot
    plt.figure(figsize=(10, 8))
    unique_classes = umap_df[outcome_col].unique() if outcome_col in umap_df else []

    # Custom color palette if number of classes <= 3
    custom_palette = ["#ff7f00", "#e377c2", "#17becf"][:len(unique_classes)] if len(unique_classes) <= 3 else "husl"

    sns.scatterplot(
        x='UMAP-1', y='UMAP-2',
        hue=umap_df[outcome_col] if outcome_col in umap_df else None,
        palette=custom_palette,
        data=umap_df,
        s=60, edgecolor='k', alpha=0.9, linewidth=0.3
    )

    plt.title(title, fontsize=16)
    plt.xlabel("UMAP Dimension 1", fontsize=13)
    plt.ylabel("UMAP Dimension 2", fontsize=13)
    if outcome_col in umap_df:
        plt.legend(title=outcome_col, fontsize=11, title_fontsize=12, loc='best')
    else:
        plt.legend([], [], frameon=False)

    # Save plot
    os.makedirs(filepath, exist_ok=True)
    save_path = os.path.join(filepath, f'umap_{classifier_name}_{sheet}_{num_features}_features.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"UMAP plot saved to {save_path}")


import os
import pandas as pd
import numpy as np
import umap
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.preprocessing import StandardScaler

def plot_umap_tsne_3d(df, classifier_name, num_features, sheet="",
                      features=None, exclude_cols=None, outcome_col='',
                      title='3D Projection of Radiomics Features',
                      filepath='./plots'):
    """
    Generates 3D UMAP and t-SNE projections for selected features and saves interactive Plotly plots.

    Parameters:
    - df (pd.DataFrame): Dataset containing features and labels.
    - classifier_name (str): For naming plot files.
    - num_features (int): Number of features selected.
    - sheet (str): Sheet name or identifier (optional).
    - features (list): Columns to include (optional).
    - exclude_cols (list): Columns to exclude (optional).
    - outcome_col (str): Column used for color grouping.
    - title (str): Plot title.
    - filepath (str): Folder to save plots.
    """
    os.makedirs(filepath, exist_ok=True)

    if exclude_cols is None:
        exclude_cols = []

    if features is None:
        features = [col for col in df.columns if col not in exclude_cols + [outcome_col]]

    features = [col for col in features if col in df.columns]
    print(f"Selected features ({len(features)}): {features}")

    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df[outcome_col].astype(str) if outcome_col in df.columns else None

    # Scale features
    X_scaled = StandardScaler().fit_transform(X)

    ### UMAP ###
    reducer = umap.UMAP(n_neighbors=15, min_dist=1, n_components=3, random_state=42)
    umap_embedding = reducer.fit_transform(X_scaled)
    umap_df = pd.DataFrame(umap_embedding, columns=['UMAP-1', 'UMAP-2', 'UMAP-3'])
    umap_df[outcome_col] = y.values

    fig_umap = px.scatter_3d(umap_df, x='UMAP-1', y='UMAP-2', z='UMAP-3',
                             color=outcome_col, title=f"3D UMAP - {title}",
                             labels={"color": outcome_col})
    fig_umap.write_html(os.path.join(filepath, f'3D_UMAP_{classifier_name}_{sheet}_{num_features}.html'))
    print(f"3D UMAP plot saved to {os.path.join(filepath, f'3D_UMAP_{classifier_name}_{sheet}_{num_features}.html')}")

    ### t-SNE ###
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    tsne_embedding = tsne.fit_transform(X_scaled)
    tsne_df = pd.DataFrame(tsne_embedding, columns=['TSNE-1', 'TSNE-2', 'TSNE-3'])
    tsne_df[outcome_col] = y.values

    fig_tsne = px.scatter_3d(tsne_df, x='TSNE-1', y='TSNE-2', z='TSNE-3',
                              color=outcome_col, title=f"3D t-SNE - {title}",
                              labels={"color": outcome_col})
    fig_tsne.write_html(os.path.join(filepath, f'3D_tSNE_{classifier_name}_{sheet}_{num_features}.html'))
    print(f"3D t-SNE plot saved to {os.path.join(filepath, f'3D_tSNE_{classifier_name}_{sheet}_{num_features}.html')}")
