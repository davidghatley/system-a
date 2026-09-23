#!/usr/bin/env python3
"""Regression checks for fold calibration row alignment on synthetic inputs."""
import unittest

import numpy as np

from scripts.i3_dev_diagnostics import grouped_cv, metrics, scaled, temperature


class GroupedCalibrationAlignmentTest(unittest.TestCase):
    def test_fold_reconstruction_preserves_original_record_alignment(self):
        # Interleave two records from each group so grouped folds are not
        # contiguous in original row order. Varied labels/confidence make a
        # concatenation-vs-index-placement error observable in pooled metrics.
        group_ids = [f"g{i}" for i in range(10)] * 2
        labels = [(i * 3 + i // 4) % 6 for i in range(len(group_ids))]
        rows = [{"record_id": f"fixture-{i}", "gold_label": ("read", "search", "edit", "execute", "other_tool", "respond_or_finish")[label],
                 "metadata": {"task_group": group_ids[i]}} for i, label in enumerate(labels)]
        y = np.asarray(labels)
        logits = np.asarray([[(i * (j + 2) + j * j) % 13 / 4 for j in range(6)] for i in range(len(y))], dtype=float)
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        groups = group_ids
        result = grouped_cv(rows, y, p, groups)
        self.assertEqual(result["pooled_raw"], metrics(y, p))
        group_members = {}
        for index, group in enumerate(groups):
            group_members.setdefault(group, []).append(index)
        folds = [[] for _ in range(5)]
        fold_counts = [0] * len(folds)
        for group, indices in sorted(group_members.items(), key=lambda item: (-len(item[1]), str(item[0]))):
            fold = min(range(len(folds)), key=lambda k: (fold_counts[k], k))
            folds[fold].extend(indices)
            fold_counts[fold] += len(indices)
        reconstructed = np.full_like(p, np.nan)
        for held in folds:
            train = np.asarray([i for other in folds if other is not held for i in other])
            held = np.asarray(held)
            t = temperature(p[train], y[train])
            reconstructed[held] = scaled(p[held], t)
        self.assertEqual(result["pooled_scaled_foldwise"], metrics(y, reconstructed))
        self.assertEqual(result["group_count"], 10)


if __name__ == "__main__":
    unittest.main()
