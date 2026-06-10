"""
@file panel_utils.py
@brief Shared panel-balancing utilities for longitudinal model fitting.
"""

import numpy as np


def balance_panel(Y, T, X, groups, W=None, max_periods=None):
    """
    @brief Pad or truncate panels to equal length via last-observation carry-forward (LOCF).

    Short panels are LOCF-padded; long panels are truncated.

    @param Y            Outcomes of shape (n_total,).
    @param T            Treatments of shape (n_total,).
    @param X            Feature matrix of shape (n_total, d_x), or None.
    @param groups       Group identifiers of shape (n_total,).
    @param W            Time-varying confounders of shape (n_total, d_w), or None.
    @param max_periods  Target periods per group; defaults to median group size (clamped >= 2).
    @return Tuple (Y_bal, T_bal, X_bal, G_bal, W_bal) with balanced arrays.
    """
    unique_groups, group_starts = np.unique(groups, return_index=True)
    group_sizes = np.diff(np.append(group_starts, len(groups)))

    if max_periods is None:
        max_periods = int(np.ceil(np.median(group_sizes)))
    max_periods = max(max_periods, 2)

    n_groups = len(unique_groups)
    print(
        f"  Balancing panels: {n_groups} groups, target {max_periods} "
        f"periods (min={group_sizes.min()}, max={group_sizes.max()}, "
        f"median={np.median(group_sizes):.0f})"
    )

    Y_bal = np.empty(n_groups * max_periods)
    T_bal = np.empty(n_groups * max_periods, dtype=T.dtype)
    X_bal = np.empty((n_groups * max_periods, X.shape[1])) if X is not None else None
    W_bal = np.empty((n_groups * max_periods, W.shape[1])) if W is not None else None
    G_bal = np.empty(n_groups * max_periods, dtype=groups.dtype)

    for i, (g, start, n_g) in enumerate(
        zip(unique_groups, group_starts, group_sizes)
    ):
        dst = slice(i * max_periods, (i + 1) * max_periods)
        n_copy = min(n_g, max_periods)
        src_end = start + n_copy
        n_pad = max_periods - n_copy

        Y_bal[dst][:n_copy] = Y[start:src_end]
        T_bal[dst][:n_copy] = T[start:src_end]
        if X is not None:
            X_bal[dst][:n_copy] = X[start:src_end]
        if W is not None:
            W_bal[dst][:n_copy] = W[start:src_end]
        G_bal[dst] = g

        if n_pad > 0:
            last = src_end - 1
            Y_bal[dst][n_copy:] = Y[last]
            T_bal[dst][n_copy:] = T[last]
            if X is not None:
                X_bal[dst][n_copy:] = X[last]
            if W is not None:
                W_bal[dst][n_copy:] = W[last]

    return Y_bal, T_bal, X_bal, G_bal, W_bal
