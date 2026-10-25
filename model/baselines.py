"""Small-data reference models with train-only PCA of inputs and output fields."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from model.evaluate import score_predictions


class MeanPredictor:
    def __init__(self, train):
        self.fields = train["fields"].mean(0)
        self.k = float(train["k_eff"].mean())

    def predict(self, cases):
        n = len(cases["x"])
        fields = np.broadcast_to(self.fields, (n, *self.fields.shape)).copy()
        fields[:, :2] *= cases["mask"][:, None]
        fields[:, 2] *= cases["fuel_mask"]
        return {"flux": fields[:, :2], "power": fields[:, 2], "k_eff": np.full(n, self.k)}


class PODBasis:
    def __init__(self, scaler, input_modes=64, output_modes=40, seed=42):
        self.scaler = scaler
        self.input_modes = input_modes
        self.output_modes = output_modes
        self.seed = seed

    def fit(self, train):
        start = time.perf_counter()
        x = self.scaler.encode_x(train["x"], train["mask"]).reshape(len(train["x"]), -1)
        y = self.scaler.encode_fields(train["fields"], train["mask"], train["fuel_mask"])
        self.field_shape = y.shape[1:]
        self.x_pca = PCA(
            n_components=min(self.input_modes, len(x) - 1, x.shape[1]),
            svd_solver="randomized",
            random_state=self.seed,
        )
        self.y_pca = PCA(
            n_components=min(self.output_modes, len(y) - 1, y[0].size),
            whiten=True,
            svd_solver="randomized",
            random_state=self.seed,
        )
        x_reduced = self.x_pca.fit_transform(x)
        y_reduced = self.y_pca.fit_transform(y.reshape(len(y), -1))
        self.fit_seconds = time.perf_counter() - start
        return x_reduced, y_reduced

    def inputs(self, cases):
        x = self.scaler.encode_x(cases["x"], cases["mask"]).reshape(len(cases["x"]), -1)
        return self.x_pca.transform(x)

    def decode(self, coefficients, k, cases):
        fields = self.y_pca.inverse_transform(coefficients).reshape(-1, *self.field_shape)
        return self.scaler.decode(fields, k, cases["mask"], cases["fuel_mask"])

    def reconstruction(self, cases):
        y = self.scaler.encode_fields(cases["fields"], cases["mask"], cases["fuel_mask"])
        coefficients = self.y_pca.transform(y.reshape(len(y), -1))
        return self.decode(coefficients, self.scaler.encode_k(cases["k_eff"]), cases)


class ClassicalPredictor:
    def __init__(self, basis, field_model, k_model):
        self.basis = basis
        self.field_model = field_model
        self.k_model = k_model

    def predict(self, cases):
        x = self.basis.inputs(cases)
        return self.basis.decode(self.field_model.predict(x), self.k_model.predict(x), cases)


def fit_classical(name, basis, data, x_train, y_train, out, seed=42, epochs=100, patience=12):
    start = time.perf_counter()
    k_train = basis.scaler.encode_k(data["train"]["k_eff"])
    best_score = float("inf")
    best = None
    history = []
    x_val = basis.inputs(data["val"])
    if name == "ridge":
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            field_model = Ridge(alpha=alpha).fit(x_train, y_train)
            k_model = Ridge(alpha=alpha).fit(x_train, k_train)
            pred = basis.decode(field_model.predict(x_val), k_model.predict(x_val), data["val"])
            score = score_predictions(pred, data["val"], basis.scaler)["normalized_score"]
            history.append({"alpha": alpha, "val_loss": score})
            if score < best_score:
                best_score = score
                best = ClassicalPredictor(basis, field_model, k_model)
    elif name == "extra_trees":
        field_model = ExtraTreesRegressor(
            n_estimators=100, min_samples_leaf=2, n_jobs=4, random_state=seed
        ).fit(x_train, y_train)
        k_model = ExtraTreesRegressor(
            n_estimators=100, min_samples_leaf=2, n_jobs=4, random_state=seed
        ).fit(x_train, k_train)
        best = ClassicalPredictor(basis, field_model, k_model)
        best_score = score_predictions(best.predict(data["val"]), data["val"], basis.scaler)[
            "normalized_score"
        ]
    elif name == "mlp":
        options = {
            "hidden_layer_sizes": (128, 128),
            "batch_size": 64,
            "learning_rate_init": 0.001,
            "random_state": seed,
            "max_iter": 1,
        }
        field_model = MLPRegressor(**options)
        k_model = MLPRegressor(**options)
        stale = 0
        for epoch in range(epochs):
            field_model.partial_fit(x_train, y_train)
            k_model.partial_fit(x_train, k_train)
            pred = basis.decode(field_model.predict(x_val), k_model.predict(x_val), data["val"])
            score = score_predictions(pred, data["val"], basis.scaler)["normalized_score"]
            history.append({"epoch": epoch + 1, "val_loss": score})
            if score < best_score:
                best_score = score
                best = ClassicalPredictor(basis, copy.deepcopy(field_model), copy.deepcopy(k_model))
                stale = 0
            else:
                stale += 1
            if (epoch + 1) % 10 == 0:
                print(f"pca_mlp epoch {epoch + 1}: val={score:.5f}", flush=True)
            if stale >= patience:
                break
    else:
        raise ValueError(name)
    result = {
        "predictor": best,
        "normalization": asdict(basis.scaler),
        "history": history,
        "best_val_loss": best_score,
        "fit_seconds": time.perf_counter() - start + basis.fit_seconds,
        "pca_seconds_included": basis.fit_seconds,
    }
    joblib.dump(result, out)
    return best, {
        key: value for key, value in result.items() if key not in ("predictor", "normalization")
    }
