"""Pure-Python tree-ensemble predictor loaded from the JSON written by src/models/export.py.

Needs only the standard library, so the deployed web app does not ship numpy/pandas/lightgbm/xgboost.
Feature vector layout (matches the training pipeline): ordinal-encoded categoricals first, then numerics.
"""
import json
import struct


def _f32(v):
    return struct.unpack("f", struct.pack("f", v))[0]


class LiteModel:
    def __init__(self, spec):
        self.version = str(spec.get("version", "unknown"))
        self.kind = spec["kind"]
        self.cat_columns = spec["cat_columns"]
        self.num_columns = spec["num_columns"]
        self.categories = spec["categories"]
        self.hist = spec["hist"]
        self.hist_default = spec["hist_default"]
        self.base = spec["base"]
        self.strict = spec["strict"]  # xgboost splits on x < t, lightgbm on x <= t
        self.trees = spec["trees"]
        if self.strict:  # xgboost evaluates in float32: round thresholds the same way
            self.trees = [[[n[0], _f32(n[1]), n[2], n[3]] if len(n) == 4 else n for n in t] for t in self.trees]

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def hist_avg(self, train_no, station):
        return self.hist.get(f"{train_no}|{station}", self.hist_default)

    def encode(self, row):
        vec = []
        for c in self.cat_columns:
            cats = self.categories[c]
            v = row[c]
            vec.append(float(cats.index(v)) if v in cats else -1.0)
        vec.extend(float(row[c]) for c in self.num_columns)
        return [_f32(v) for v in vec] if self.strict else vec

    def predict(self, row):
        """row: dict with every cat and num column; returns the predicted delay in minutes."""
        x = self.encode(row)
        total = self.base
        strict = self.strict
        for nodes in self.trees:
            i = 0
            node = nodes[0]
            while len(node) == 4:
                f, thr, left, right = node
                go_left = x[f] < thr if strict else x[f] <= thr
                i = left if go_left else right
                node = nodes[i]
            total += node[0]
        return total
