"""
Part of Phase 7: train/validation/test split
------------------------------------------------
WHY A STRATIFIED RANDOM SPLIT: fraud is severely imbalanced (~8.74%
positive in this dataset). Stratifying on the target ensures train/val/
test each preserve that rate, so none of them is accidentally easier or
harder than the others just from random imbalance in the split itself —
the same reasoning used in the companion churn and attrition projects,
and for the same reason (a rare positive class is exactly the case where
an unlucky random split matters most). Unlike the sales-forecasting
project, this dataset has no time dimension (it's a simulated,
cross-sectional set of transactions with no timestamp), so a random
split is the correct choice here rather than a chronological one.
"""

from sklearn.model_selection import train_test_split


def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
