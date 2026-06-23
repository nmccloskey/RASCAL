import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# -------------------------------
# Load data
# -------------------------------
wab_data_path = os.path.join("..", "July2025_PrepSpreadsheets", "MASTERDiscourseInter_DATA_2025-02-19_1542.csv")
wab_df = pd.read_csv(wab_data_path)

trel_path = os.path.join("rascal_d_output_250910_0901", "TranscriptionReliabilityAnalysis", "TranscriptionReliabilityAnalysis.xlsx")
trel_df = pd.read_excel(trel_path)

# Aggregate transcription reliability over participant
graph_df = trel_df.groupby(by="study_id").agg(
    mean_ls=("LevenshteinSimilarity", "mean"),
    num_stim=("OrgFile", "count")
).reset_index()

# Merge clinical metrics
graph_df = graph_df.merge(
    wab_df[["study_id", "wabaq", "wabseverity", "wabaphasiasyndrome", "namtotscore1"]],
    how="inner", on="study_id"
)

# -------------------------------
# Helper function to plot
# -------------------------------
def plot_relationship(df, x_col, x_label, title, filename=None, filter_positive=False):
    data = df.copy()
    if filter_positive:
        data = data[data[x_col] > 0]

    # Drop rows with missing data
    data = data.dropna(subset=[x_col, "mean_ls"])

    plt.figure(figsize=(10, 6))

    # Scatter
    plt.scatter(
        data[x_col], data["mean_ls"],
        s=data["num_stim"] * 20,
        alpha=0.7,
        c="tab:blue",
        edgecolors="black",
        linewidth=0.5,
    )

    # Regression line + CI (only if we have >=2 unique points)
    if data[x_col].nunique() > 1:
        sns.regplot(
            data=data,
            x=x_col, y="mean_ls",
            scatter=False,
            ci=95,
            line_kws={"color": "black", "lw": 2, "alpha": 0.8}
        )

        slope, intercept, r_value, p_value, std_err = linregress(data[x_col], data["mean_ls"])
        stats_text = (
            f"y = {intercept:.2f} + {slope:.2f}x\n"
            f"$R^2$ = {r_value**2:.3f}\n"
            f"p = {p_value:.3g}\n"
            f"SE = {std_err:.3f}"
        )
    else:
        stats_text = "Regression not computed:\ninsufficient variation"

    # Annotation box
    plt.gca().text(
        0.05, 0.05, stats_text,
        transform=plt.gca().transAxes,
        fontsize=13,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8)
    )

    # Labels
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel("Mean Levenshtein Similarity", fontsize=14)
    plt.title(title, fontsize=18)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

# -------------------------------
# Generate plots
# -------------------------------
sns.set_theme(style="whitegrid")

# WAB-AQ plot
plot_relationship(
    graph_df,
    x_col="wabaq",
    x_label="WAB-AQ (Aphasia Severity)",
    title="Relationship between WAB-AQ and Autotranscription Accuracy",
    filename="wab_vs_accuracy.png",
    filter_positive=True
)

# CAT Naming Score plot
plot_relationship(
    graph_df,
    x_col="namtotscore1",
    x_label="CAT Total Naming Score",
    title="Relationship between CAT Naming Score and Autotranscription Accuracy",
    filename="cat_naming_vs_accuracy.png",
    filter_positive=False
)

# Prep summary table
sum_df = trel_df.merge(
    wab_df[["study_id", "wabaq", "wabseverity", "wabaphasiasyndrome", "namtotscore1"]],
    how="inner", on="study_id"
)
cols = ["Measure", "All Pairs", "WABAQ", "CAT Naming"]
all_transcs = pd.DataFrame(sum_df["LevenshteinSimilarity"].describe())
wab_transcs = pd.DataFrame(sum_df[sum_df["wabaq"]>0]["LevenshteinSimilarity"].describe())
cat_transcs = pd.DataFrame(sum_df[~sum_df["namtotscore1"].isnull()]["LevenshteinSimilarity"].describe())

desc_df = pd.concat([all_transcs, wab_transcs, cat_transcs], axis=1)