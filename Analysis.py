import pandas as pd
import numpy as np
import pingouin as pg
from scipy.stats import shapiro, levene
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# ============================================================
# LOAD DATA
# ============================================================

CSV_FILE = "analysis_dataset.csv"

df = pd.read_csv(CSV_FILE)

print(f"\nLoaded {len(df)} rows")
print(f"Participants: {df['participant_id'].nunique()}")
print(f"Statements: {df['statement_id'].nunique()}")

# ============================================================
# CLEAN DATA
# ============================================================

df["label_visible"] = df["label_visible"].map(
    {
        0: "Unlabeled",
        1: "Labeled"
    }
)

df["actual_source"] = (
    df["actual_source"]
    .astype(str)
    .str.lower()
)

df["known_truth"] = (
    df["known_truth"]
    .astype(str)
)

# ============================================================
# CREATE THE 4 REPEATED-MEASURE CONDITIONS
#
# True_AI
# False_AI
# True_Human
# False_Human
#
# Matches the G*Power setup:
# "4 repeated measurements"
# ============================================================

df["condition"] = (
    np.where(
        df["known_truth"].str.lower() == "true",
        "True_",
        "False_"
    )
    +
    np.where(
        df["actual_source"] == "ai",
        "AI",
        "Human"
    )
)

print("\nCondition counts:")
print(df["condition"].value_counts())

# ============================================================
# AGGREGATE TO PARTICIPANT × CONDITION
# ============================================================

agg = (
    df.groupby(
        [
            "participant_id",
            "label_visible",
            "condition"
        ]
    )
    .agg(
        accuracy=("correct", "mean"),
        confidence=("confidence", "mean"),
        trustworthiness=("trustworthiness", "mean")
    )
    .reset_index()
)

print("\nAggregated dataset shape:")
print(agg.shape)

# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================


condition_counts = (
    agg.groupby("participant_id")["condition"]
       .nunique()
)

complete = (
    agg.groupby("participant_id")["condition"]
       .nunique()
)

print(len(complete))
complete = complete[complete == 4]
print("Participants retained:", len(complete))
complete_ids = complete[complete == 4].index
agg = agg[agg["participant_id"].isin(complete_ids)]
print(df.columns.tolist())
print(df.head())


print("\n===================================================")
print("DESCRIPTIVE STATISTICS")
print("===================================================")

desc = (
    agg.groupby(
        ["label_visible", "condition"]
    )
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        confidence_mean=("confidence", "mean"),
        confidence_sd=("confidence", "std"),
        trust_mean=("trustworthiness", "mean"),
        trust_sd=("trustworthiness", "std")
    )
)

print(desc)

desc.to_csv("descriptive_statistics.csv")

# ============================================================
# ASSUMPTION TESTS
# ============================================================

print("\n===================================================")
print("SHAPIRO-WILK NORMALITY TESTS")
print("===================================================")

for dv in ["accuracy", "confidence", "trustworthiness"]:

    stat, p = shapiro(
        agg[dv].dropna()
    )

    print(
        f"{dv:<20} W={stat:.4f} p={p:.4f}"
    )

# ============================================================
# LEVENE TESTS
# ============================================================

print("\n===================================================")
print("LEVENE HOMOGENEITY TESTS")
print("===================================================")


agg = agg.dropna(
    subset=["accuracy", "confidence", "trustworthiness"]
)

for dv in ["accuracy", "confidence", "trustworthiness"]:

    labeled = agg.loc[
        agg["label_visible"] == "Labeled",
        dv
    ]

    unlabeled = agg.loc[
        agg["label_visible"] == "Unlabeled",
        dv
    ]

    stat, p = levene(
        labeled,
        unlabeled
    )

    print(
        f"{dv:<20} F={stat:.4f} p={p:.4f}"
    )

# ============================================================
# MAUCHLY'S TEST OF SPHERICITY
# ============================================================

print("\n===================================================")
print("MAUCHLY'S TEST OF SPHERICITY")
print("===================================================")

for dv in ["accuracy", "confidence", "trustworthiness"]:

    print(f"\n{dv.upper()}")

    try:

        wide = agg.pivot(
            index="participant_id",
            columns="condition",
            values=dv
        )

        result = pg.sphericity(wide)

        print(result)

    except Exception as e:
        print("Could not compute:", e)

# ============================================================
# MIXED ANOVA - ACCURACY
# ============================================================

print("\n===================================================")
print("MIXED ANOVA - ACCURACY")
print("===================================================")

anova_accuracy = pg.mixed_anova(
    data=agg,
    dv="accuracy",
    within="condition",
    between="label_visible",
    subject="participant_id"
)

print(anova_accuracy)

anova_accuracy.to_csv(
    "anova_accuracy.csv",
    index=False
)

# ============================================================
# MIXED ANOVA - CONFIDENCE
# ============================================================

print("\n===================================================")
print("MIXED ANOVA - CONFIDENCE")
print("===================================================")

anova_confidence = pg.mixed_anova(
    data=agg,
    dv="confidence",
    within="condition",
    between="label_visible",
    subject="participant_id"
)

print(anova_confidence)

anova_confidence.to_csv(
    "anova_confidence.csv",
    index=False
)

# ============================================================
# MIXED ANOVA - TRUSTWORTHINESS
# ============================================================

print("\n===================================================")
print("MIXED ANOVA - TRUSTWORTHINESS")
print("===================================================")

anova_trust = pg.mixed_anova(
    data=agg,
    dv="trustworthiness",
    within="condition",
    between="label_visible",
    subject="participant_id"
)

print(anova_trust)

anova_trust.to_csv(
    "anova_trustworthiness.csv",
    index=False
)

# ============================================================
# POST-HOC TESTS
# ============================================================

print("\n===================================================")
print("POST-HOC TESTS (BONFERRONI)")
print("===================================================")

posthoc_accuracy = pg.pairwise_tests(
    data=agg,
    dv="accuracy",
    within="condition",
    between="label_visible",
    subject="participant_id",
    padjust="bonf",
    effsize="hedges"
)

posthoc_confidence = pg.pairwise_tests(
    data=agg,
    dv="confidence",
    within="condition",
    between="label_visible",
    subject="participant_id",
    padjust="bonf",
    effsize="hedges"
)

posthoc_trust = pg.pairwise_tests(
    data=agg,
    dv="trustworthiness",
    within="condition",
    between="label_visible",
    subject="participant_id",
    padjust="bonf",
    effsize="hedges"
)

print("\nAccuracy Post-hoc")
print(posthoc_accuracy.head())

print("\nConfidence Post-hoc")
print(posthoc_confidence.head())

print("\nTrustworthiness Post-hoc")
print(posthoc_trust.head())

posthoc_accuracy.to_csv(
    "posthoc_accuracy.csv",
    index=False
)

posthoc_confidence.to_csv(
    "posthoc_confidence.csv",
    index=False
)

posthoc_trust.to_csv(
    "posthoc_trustworthiness.csv",
    index=False
)

# ============================================================
# EFFECT SIZE SUMMARY
# ============================================================

print("\n===================================================")
print("EFFECT SIZES (Partial Eta Squared)")
print("===================================================")

for name, result in [
    ("Accuracy", anova_accuracy),
    ("Confidence", anova_confidence),
    ("Trustworthiness", anova_trust)
]:
    print(f"\n{name}")
    print(anova_accuracy.columns)
    print(anova_confidence.columns)
    print(anova_trust.columns)
    print(
        result[
            [
                "Source",
                "F",
                "p_unc",
                "p_GG_corr",
                "np2"
            ]
        ]
    )

# ============================================================
# H2 ANALYSIS
# UNLABELED ONLY
# AI VS HUMAN TRUSTWORTHINESS
# ============================================================

print("\n===================================================")
print("H2 TEST: UNLABELED TRUSTWORTHINESS")
print("AI vs HUMAN")
print("===================================================")

unlabeled = agg[
    agg["label_visible"] == "Unlabeled"
].copy()

# Collapse conditions into source only

unlabeled["source"] = np.where(
    unlabeled["condition"].str.contains("AI"),
    "AI",
    "Human"
)

h2_desc = (
    unlabeled
    .groupby("source")
    ["trustworthiness"]
    .agg(["mean", "std", "count"])
)

print("\nDescriptive statistics:")
print(h2_desc)

h2_test = pg.pairwise_tests(
    data=unlabeled,
    dv="trustworthiness",
    within="source",
    subject="participant_id",
    padjust="bonf",
    effsize="hedges"
)

print("\nAI vs Human trustworthiness:")
print(h2_test)

h2_test.to_csv(
    "h2_unlabeled_trustworthiness.csv",
    index=False
)

print("\n===================================================")
print("ANALYSIS COMPLETE")
print("===================================================")

print("\nFiles written:")
print(" - descriptive_statistics.csv")
print(" - anova_accuracy.csv")
print(" - anova_confidence.csv")
print(" - anova_trustworthiness.csv")
print(" - posthoc_accuracy.csv")
print(" - posthoc_confidence.csv")
print(" - posthoc_trustworthiness.csv")

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import scipy.stats as stats

dvs = ["accuracy", "confidence", "trustworthiness"]

conditions = agg["condition"].unique()
#
# for dv in dvs:
#     for cond in conditions:
#
#         subset = agg[agg["condition"] == cond][dv].dropna()
#
#         plt.figure(figsize=(6, 6))
#
#         stats.probplot(subset, dist="norm", plot=plt)
#
#         plt.title(f"Q-Q Plot: {dv} ({cond})")
#
#         plt.grid(True)
#
#         plt.show()