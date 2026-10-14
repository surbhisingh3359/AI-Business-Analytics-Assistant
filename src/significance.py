"""
Statistical support for the insight layer.

The analysis pipeline previously graded a finding by the size of the gap
alone, so a 60% difference between two groups of two rows was reported as
"high significance". This module supplies the missing half of the claim:
whether a difference is distinguishable from noise, how large it is on a
standardised scale, and how many observations back it.

Every public function returns the same shaped verdict so the insight and
decision layers can consume any test without special-casing it:

    {
        "test":             name of the test that was run
        "statistic":        the test statistic
        "p_value":          uncorrected p-value
        "effect_size":      standardised effect size
        "effect_metric":    which effect size was used
        "effect_label":     negligible | small | medium | large
        "n":                observations the test used
        "significant":      p_value < ALPHA
        "underpowered":     a group was too small to trust
        "confidence":       high | medium | low | insufficient
        "note":             caveat to show alongside the finding, or ""
        "robustness_test":  distribution-free cross-check, when available
        "robustness_p_value": its p-value
    }

Findings should be read through ``confidence``, which requires statistical
support, a non-negligible effect, and adequate sample size together. A
result can be significant and still carry low confidence.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# =========================================================
# CONFIGURATION
# =========================================================

ALPHA = 0.05

# Below this, a group's mean is too unstable to report as a difference.
MIN_GROUP_SIZE = 5

# Below this, no test can be run at all.
MIN_TESTABLE_GROUP_SIZE = 2

# Standard Cohen thresholds for each effect-size family.
EFFECT_THRESHOLDS = {
    "cohens_d": (0.2, 0.5, 0.8),
    "eta_squared": (0.01, 0.06, 0.14),
    "cramers_v": (0.1, 0.3, 0.5),
    "pearson_r": (0.1, 0.3, 0.5),
}

EFFECT_LABELS = (
    "negligible",
    "small",
    "medium",
    "large",
)


# =========================================================
# SHARED HELPERS
# =========================================================

def _finite(
    values: Sequence[float],
) -> np.ndarray:
    """Coerce a sequence to a finite float array."""

    array = pd.to_numeric(
        pd.Series(list(values)),
        errors="coerce",
    ).to_numpy(dtype="float64")

    return array[np.isfinite(array)]


def _clean_float(
    value: Any,
) -> float | None:
    """Return a JSON-safe float, or None for nan/inf."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(number):
        return None

    return number


def label_effect(
    effect_size: float | None,
    effect_metric: str,
) -> str:
    """Map an effect size onto a negligible/small/medium/large label."""

    if effect_size is None:
        return "unknown"

    thresholds = EFFECT_THRESHOLDS.get(effect_metric)

    if thresholds is None:
        return "unknown"

    magnitude = abs(effect_size)

    small, medium, large = thresholds

    if magnitude >= large:
        return "large"

    if magnitude >= medium:
        return "medium"

    if magnitude >= small:
        return "small"

    return "negligible"


def grade_confidence(
    p_value: float | None,
    effect_label: str,
    smallest_group: int,
    underpowered: bool,
) -> str:
    """
    Grade a finding on statistics rather than magnitude alone.

    High confidence requires all three of a clear p-value, an effect that
    is at least small, and enough rows in every group. Any one of those
    failing pulls the grade down.
    """

    if p_value is None or smallest_group < MIN_TESTABLE_GROUP_SIZE:
        return "insufficient"

    if effect_label in {"negligible", "unknown"}:
        return "low"

    if p_value >= ALPHA:
        return "low"

    # A significant result resting on two or three rows is not reportable,
    # however large the effect looks.
    if smallest_group < 3:
        return "low"

    if underpowered or p_value >= 0.01 or effect_label == "small":
        return "medium"

    return "high"


def _verdict(
    test: str,
    statistic: Any,
    p_value: Any,
    effect_size: Any,
    effect_metric: str,
    n: int,
    smallest_group: int,
    note: str = "",
    robustness_test: str = "",
    robustness_p_value: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    """Assemble the standard verdict dictionary."""

    clean_p = _clean_float(p_value)
    clean_effect = _clean_float(effect_size)
    effect_label = label_effect(clean_effect, effect_metric)

    underpowered = smallest_group < MIN_GROUP_SIZE

    notes = [note] if note else []

    if underpowered:
        notes.append(
            f"Smallest group has {smallest_group} row(s); "
            f"at least {MIN_GROUP_SIZE} are needed for a stable estimate."
        )

    verdict = {
        "test": test,
        "statistic": _clean_float(statistic),
        "p_value": clean_p,
        "effect_size": clean_effect,
        "effect_metric": effect_metric,
        "effect_label": effect_label,
        "n": int(n),
        "smallest_group": int(smallest_group),
        "significant": bool(clean_p is not None and clean_p < ALPHA),
        "underpowered": underpowered,
        "confidence": grade_confidence(
            clean_p,
            effect_label,
            smallest_group,
            underpowered,
        ),
        "note": " ".join(notes),
    }

    if robustness_test:
        verdict["robustness_test"] = robustness_test
        verdict["robustness_p_value"] = _clean_float(robustness_p_value)

    verdict.update(extra)

    return verdict


def _untestable(
    test: str,
    effect_metric: str,
    n: int,
    smallest_group: int,
    note: str,
) -> dict[str, Any]:
    """Return a verdict for a comparison that could not be tested."""

    return {
        "test": test,
        "statistic": None,
        "p_value": None,
        "effect_size": None,
        "effect_metric": effect_metric,
        "effect_label": "unknown",
        "n": int(n),
        "smallest_group": int(smallest_group),
        "significant": False,
        "underpowered": True,
        "confidence": "insufficient",
        "note": note,
    }


# =========================================================
# EFFECT SIZES
# =========================================================

def cohens_d(
    first: Sequence[float],
    second: Sequence[float],
) -> float | None:
    """Standardised mean difference between two samples."""

    a = _finite(first)
    b = _finite(second)

    if a.size < 2 or b.size < 2:
        return None

    pooled_variance = (
        (a.size - 1) * a.var(ddof=1)
        + (b.size - 1) * b.var(ddof=1)
    ) / (a.size + b.size - 2)

    if pooled_variance <= 0:
        return None

    return float(
        (a.mean() - b.mean()) / np.sqrt(pooled_variance)
    )


def eta_squared(
    f_statistic: float,
    df_between: int,
    df_within: int,
) -> float | None:
    """Share of variance in the metric explained by the grouping."""

    if df_between <= 0 or df_within <= 0:
        return None

    numerator = f_statistic * df_between

    denominator = numerator + df_within

    if denominator <= 0:
        return None

    return float(numerator / denominator)


def cramers_v(
    chi_square: float,
    n: int,
    rows: int,
    columns: int,
) -> float | None:
    """Association strength for a contingency table."""

    smaller_dimension = min(rows, columns) - 1

    if n <= 0 or smaller_dimension <= 0:
        return None

    return float(
        np.sqrt(chi_square / (n * smaller_dimension))
    )


# =========================================================
# GROUP DIFFERENCES
# =========================================================

def test_group_difference(
    dataframe: pd.DataFrame,
    category: str,
    metric: str,
) -> dict[str, Any]:
    """
    Test whether a metric genuinely differs across the groups of a column.

    Two groups are compared with Welch's t-test, which does not assume
    equal variances, and cross-checked with Mann-Whitney U. Three or more
    groups use one-way ANOVA, cross-checked with Kruskal-Wallis. The
    cross-check is reported rather than used to switch tests, so a reader
    can see when the parametric result rests on a shaky assumption.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    for column in (category, metric):

        if column not in dataframe.columns:
            raise ValueError(
                f"Column not found in dataset: {column}"
            )

    frame = pd.DataFrame(
        {
            "group": dataframe[category],
            "value": pd.to_numeric(
                dataframe[metric],
                errors="coerce",
            ),
        }
    ).dropna()

    samples: list[np.ndarray] = []
    group_sizes: dict[str, int] = {}

    for name, chunk in frame.groupby("group", observed=True):

        values = _finite(chunk["value"])

        if values.size == 0:
            continue

        samples.append(values)
        group_sizes[str(name)] = int(values.size)

    total = int(sum(group_sizes.values()))
    smallest = min(group_sizes.values()) if group_sizes else 0

    if len(samples) < 2:
        return _untestable(
            test="none",
            effect_metric="cohens_d",
            n=total,
            smallest_group=smallest,
            note=(
                "Fewer than two groups have usable values, "
                "so no comparison is possible."
            ),
        ) | {"group_sizes": group_sizes}

    if smallest < MIN_TESTABLE_GROUP_SIZE:
        return _untestable(
            test="none",
            effect_metric="cohens_d",
            n=total,
            smallest_group=smallest,
            note=(
                f"A group has only {smallest} row(s), which is below the "
                f"{MIN_TESTABLE_GROUP_SIZE} needed to estimate variance."
            ),
        ) | {"group_sizes": group_sizes}

    if all(np.ptp(sample) == 0 for sample in samples):
        return _untestable(
            test="none",
            effect_metric="cohens_d",
            n=total,
            smallest_group=smallest,
            note="Every group is constant, so no variance can be tested.",
        ) | {"group_sizes": group_sizes}

    if len(samples) == 2:

        first, second = samples

        parametric = stats.ttest_ind(
            first,
            second,
            equal_var=False,
        )

        try:
            nonparametric_p = stats.mannwhitneyu(
                first,
                second,
                alternative="two-sided",
            ).pvalue
        except ValueError:
            nonparametric_p = None

        return _verdict(
            test="Welch t-test",
            statistic=parametric.statistic,
            p_value=parametric.pvalue,
            effect_size=cohens_d(first, second),
            effect_metric="cohens_d",
            n=total,
            smallest_group=smallest,
            robustness_test="Mann-Whitney U",
            robustness_p_value=nonparametric_p,
            group_sizes=group_sizes,
        )

    result = stats.f_oneway(*samples)

    df_between = len(samples) - 1
    df_within = total - len(samples)

    try:
        nonparametric_p = stats.kruskal(*samples).pvalue
    except ValueError:
        nonparametric_p = None

    return _verdict(
        test="One-way ANOVA",
        statistic=result.statistic,
        p_value=result.pvalue,
        effect_size=eta_squared(
            float(result.statistic),
            df_between,
            df_within,
        ),
        effect_metric="eta_squared",
        n=total,
        smallest_group=smallest,
        robustness_test="Kruskal-Wallis",
        robustness_p_value=nonparametric_p,
        group_sizes=group_sizes,
    )


# =========================================================
# CATEGORICAL ASSOCIATION
# =========================================================

def test_association(
    dataframe: pd.DataFrame,
    first_column: str,
    second_column: str,
) -> dict[str, Any]:
    """
    Test whether two categorical columns are associated.

    Uses a chi-square test of independence with Cramer's V as the effect
    size, and reports when the expected-frequency assumption is violated
    rather than quietly returning the p-value anyway.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    for column in (first_column, second_column):

        if column not in dataframe.columns:
            raise ValueError(
                f"Column not found in dataset: {column}"
            )

    table = pd.crosstab(
        dataframe[first_column],
        dataframe[second_column],
    )

    total = int(table.to_numpy().sum())

    if table.shape[0] < 2 or table.shape[1] < 2:
        return _untestable(
            test="none",
            effect_metric="cramers_v",
            n=total,
            smallest_group=0,
            note=(
                "A contingency test needs at least two levels "
                "in both columns."
            ),
        )

    chi_square, p_value, dof, expected = stats.chi2_contingency(table)

    small_expected = int((expected < 5).sum())

    note = ""

    if small_expected:
        note = (
            f"{small_expected} of {expected.size} expected cell counts are "
            "below 5, so the chi-square approximation is unreliable here."
        )

    smallest_cell = int(table.to_numpy().min())

    verdict = _verdict(
        test="Chi-square test of independence",
        statistic=chi_square,
        p_value=p_value,
        effect_size=cramers_v(
            float(chi_square),
            total,
            table.shape[0],
            table.shape[1],
        ),
        effect_metric="cramers_v",
        n=total,
        smallest_group=smallest_cell,
        note=note,
        degrees_of_freedom=int(dof),
        table_shape=list(table.shape),
    )

    if small_expected:
        verdict["confidence"] = (
            "low"
            if verdict["confidence"] == "high"
            else verdict["confidence"]
        )

    return verdict


# =========================================================
# CORRELATION
# =========================================================

def test_correlation(
    dataframe: pd.DataFrame,
    first_column: str,
    second_column: str,
) -> dict[str, Any]:
    """
    Test a linear relationship between two numeric columns.

    Pearson's r is reported with its p-value and the pairwise-complete
    sample size, and Spearman's rho is the distribution-free cross-check.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    for column in (first_column, second_column):

        if column not in dataframe.columns:
            raise ValueError(
                f"Column not found in dataset: {column}"
            )

    pairs = pd.DataFrame(
        {
            "x": pd.to_numeric(
                dataframe[first_column],
                errors="coerce",
            ),
            "y": pd.to_numeric(
                dataframe[second_column],
                errors="coerce",
            ),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()

    n = int(len(pairs))

    if n < 3:
        return _untestable(
            test="none",
            effect_metric="pearson_r",
            n=n,
            smallest_group=n,
            note=(
                "At least three paired observations are needed "
                "to test a correlation."
            ),
        )

    if pairs["x"].nunique() < 2 or pairs["y"].nunique() < 2:
        return _untestable(
            test="none",
            effect_metric="pearson_r",
            n=n,
            smallest_group=n,
            note=(
                "One of the columns is constant, "
                "so no relationship can be measured."
            ),
        )

    pearson = stats.pearsonr(pairs["x"], pairs["y"])
    spearman = stats.spearmanr(pairs["x"], pairs["y"])

    return _verdict(
        test="Pearson correlation",
        statistic=pearson.statistic,
        p_value=pearson.pvalue,
        effect_size=pearson.statistic,
        effect_metric="pearson_r",
        n=n,
        smallest_group=n,
        robustness_test="Spearman rank correlation",
        robustness_p_value=spearman.pvalue,
    )


# =========================================================
# MULTIPLE COMPARISONS
# =========================================================

def adjust_p_values(
    p_values: Sequence[float | None],
) -> list[float | None]:
    """
    Benjamini-Hochberg false-discovery-rate adjustment.

    The pipeline tests every viable category-by-metric pair, so a handful
    of "significant" results is expected by chance alone. Adjusting turns
    the raw p-values into q-values that can be compared to ALPHA directly.
    None values pass through untouched and are excluded from the count.
    """

    indexed = [
        (index, value)
        for index, value in enumerate(p_values)
        if value is not None and np.isfinite(value)
    ]

    adjusted: list[float | None] = [None] * len(p_values)

    total = len(indexed)

    if total == 0:
        return adjusted

    indexed.sort(key=lambda item: item[1])

    running_minimum = 1.0

    for rank in range(total, 0, -1):

        index, raw = indexed[rank - 1]

        candidate = min(1.0, raw * total / rank)

        running_minimum = min(running_minimum, candidate)

        adjusted[index] = float(running_minimum)

    return adjusted


def apply_fdr_correction(
    results: Sequence[dict[str, Any]],
    p_key: str = "p_value",
    q_key: str = "q_value",
) -> list[dict[str, Any]]:
    """
    Add FDR-adjusted q-values to a family of test verdicts in place.

    Confidence is re-graded against the q-value so a finding that only
    looked significant because many comparisons were run is demoted.
    """

    corrected = [dict(result) for result in results]

    adjusted = adjust_p_values(
        [result.get(p_key) for result in corrected]
    )

    for result, q_value in zip(corrected, adjusted):

        result[q_key] = q_value

        if q_value is None:
            continue

        result["significant_after_correction"] = bool(q_value < ALPHA)

        result["confidence"] = grade_confidence(
            q_value,
            result.get("effect_label", "unknown"),
            int(result.get("smallest_group", 0) or 0),
            bool(result.get("underpowered", False)),
        )

    return corrected


# =========================================================
# REPORTING
# =========================================================

def format_p_value(
    p_value: float | None,
) -> str:
    """Render a p-value the way a report would."""

    if p_value is None:
        return "p not available"

    if p_value < 0.001:
        return "p < 0.001"

    return f"p = {p_value:.3f}"


def describe_verdict(
    verdict: dict[str, Any],
) -> str:
    """
    Summarise a verdict as one clause suitable for an insight sentence.

    Reads the corrected q-value when one is present, because that is the
    number a reader should judge the finding by.
    """

    if not isinstance(verdict, dict):
        return ""

    if verdict.get("p_value") is None:
        note = verdict.get("note") or "not statistically testable"
        return f"Not tested: {note}"

    q_value = verdict.get("q_value")

    p_for_report = (
        q_value
        if q_value is not None
        else verdict.get("p_value")
    )

    label = "q" if q_value is not None else "p"

    rendered = format_p_value(p_for_report).replace("p", label, 1)

    effect = verdict.get("effect_label", "unknown")
    metric = verdict.get("effect_metric", "")
    size = verdict.get("effect_size")

    parts = [
        f"{verdict.get('test', 'test')}: {rendered}",
        f"{effect} effect",
    ]

    if size is not None and metric:
        parts[-1] = f"{effect} effect ({metric} = {size:.2f})"

    parts.append(f"n = {verdict.get('n', 0):,}")

    return ", ".join(parts) + "."
