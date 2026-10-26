"""Association rule mining: which item combinations imply fraud?"""
from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

from . import config

FRAUD_ITEM = "fraud=YES"


def run(baskets: pd.DataFrame, algorithm: str = "fpgrowth") -> pd.DataFrame:
    """Mine rules of the form {attributes} -> {fraud=YES}.

    FP-Growth gives the same frequent itemsets as Apriori but is much faster
    on a million-row table; pass algorithm="apriori" to use Apriori itself.
    """
    onehot = pd.get_dummies(baskets, prefix="", prefix_sep="").astype(bool)
    onehot = onehot.drop(columns=["fraud=NO"], errors="ignore")

    if algorithm == "apriori":
        from mlxtend.frequent_patterns import apriori
        itemsets = apriori(onehot, min_support=config.MIN_SUPPORT,
                           use_colnames=True, max_len=config.MAX_RULE_LEN, low_memory=True)
    else:
        itemsets = fpgrowth(onehot, min_support=config.MIN_SUPPORT,
                            use_colnames=True, max_len=config.MAX_RULE_LEN)

    rules = association_rules(itemsets, metric="confidence",
                              min_threshold=config.MIN_CONFIDENCE)
    rules = rules[rules["consequents"] == frozenset({FRAUD_ITEM})]
    rules = rules[rules["lift"] >= config.MIN_LIFT]
    rules = rules.sort_values(["lift", "confidence"], ascending=False)

    out = rules[["antecedents", "support", "confidence", "lift"]].copy()
    out["antecedents"] = out["antecedents"].apply(lambda s: " AND ".join(sorted(s)))
    out = out.rename(columns={"antecedents": "if"})
    out["then"] = FRAUD_ITEM
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.REPORTS_DIR / "fraud_rules.csv", index=False)
    return out.reset_index(drop=True)
