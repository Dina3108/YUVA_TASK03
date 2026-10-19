# Week 3 - A Simple Hypothesis Test
# Question: do the two BMI-feature groups have different average target values?

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_diabetes

output_folder = Path("week3_outputs")
output_folder.mkdir(exist_ok=True)

data = load_diabetes(as_frame=True).frame
middle_bmi = data["bmi"].median()

# Use the median only to make two groups for this exercise.
lower_group = data.loc[data["bmi"] < middle_bmi, "target"]
upper_group = data.loc[data["bmi"] >= middle_bmi, "target"]

print("Average target for the lower BMI-feature group:",
      round(lower_group.mean(), 2))
print("Average target for the at-or-above-median group:",
      round(upper_group.mean(), 2))

# H0: the two group averages are equal.
# H1: the two group averages are different.
test_result = stats.ttest_ind(lower_group, upper_group, equal_var=False)
difference = upper_group.mean() - lower_group.mean()

# Calculate a 95% confidence interval for the difference in means.
standard_error = np.sqrt(
    lower_group.var(ddof=1) / len(lower_group)
    + upper_group.var(ddof=1) / len(upper_group)
)
degrees = (
    lower_group.var(ddof=1) / len(lower_group)
    + upper_group.var(ddof=1) / len(upper_group)
) ** 2 / (
    (lower_group.var(ddof=1) / len(lower_group)) ** 2 / (len(lower_group) - 1)
    + (upper_group.var(ddof=1) / len(upper_group)) ** 2 / (len(upper_group) - 1)
)
margin = stats.t.ppf(0.975, degrees) * standard_error

print("\nWelch t-test result")
print("t statistic:", round(test_result.statistic, 4))
print("p-value:", test_result.pvalue)
print("Difference in means (upper - lower):", round(difference, 2))
print("95% confidence interval:",
      (round(difference - margin, 2), round(difference + margin, 2)))

if test_result.pvalue < 0.05:
    print("At the 5% level, the result is statistically significant.")
else:
    print("At the 5% level, there is not enough evidence of a difference.")

# Visual check: compare the two distributions.
import pandas as pd

group_data = pd.DataFrame({
    "BMI group": ["Below median"] * len(lower_group)
                 + ["At/above median"] * len(upper_group),
    "Target": list(lower_group) + list(upper_group)
})
plt.figure(figsize=(8, 5))
sns.boxplot(data=group_data, x="BMI group", y="Target")
plt.title("Target Values by BMI Feature Group")
plt.tight_layout()
plt.savefig(output_folder / "group_boxplot.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(data=data, x="target", bins=25, kde=True)
plt.title("Overall Target Distribution")
plt.tight_layout()
plt.savefig(output_folder / "target_histogram.png", dpi=150)
plt.close()

print("\nCharts saved in", output_folder)
