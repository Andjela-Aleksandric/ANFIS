"""
ANFIS eksperiment: predvidjanje dnevnog prinosa ISE (Istanbul Stock Exchange)
indeksa (Output_1) na osnovu prinosa tri medjunarodna trzisna indikatora
(Input_1-3) i drugog ISE prinosa (Output_2), na originalnom UCI "Istanbul
Stock Exchange" datasetu (Akbilgic et al.), istom na koji se oslanja rad
(Boyacioglu & Avci, 2010) naveden u pregledu literature

Testirane metode (hronoloski train/test split, 80/20):
  1. xanfis - AnfisRegressor      (klasican hibridni trener: gradijent + LSE)
  2. xanfis - GdAnfisRegressor    (iskljucivo gradijentni trener, RMSprop)
  3. xanfis - BioAnfisRegressor   (PSO, rojevska/bio-inspirisana optimizacija)
  4. gabrielegilardi/ANFIS        (samostalna PSO implementacija, bez torch-a)
  5. sanfis ("plain" rezim)       (PyTorch, premisa = posledica, nema state-a)
  6. sanfis (S-ANFIS rezim)       (PyTorch, Regime = predznak Input_1 kao
                                   "state"/premisa promenljiva - operacionalizacija
                                   regime-switching ideje iz Lenhard & Maringer, 2022)
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
np.random.seed(42)
SEED = 42


BASE_DIR = Path(__file__).resolve().parent


df = pd.read_csv(BASE_DIR / "ise_features.csv")
ORDINARY_FEATURES = ["Input_1", "Input_2", "Input_3", "Output_2"]
TARGET = "Output_1"

n = len(df)
n_train = int(n * 0.8)

X_all = df[ORDINARY_FEATURES].values
y_all = df[TARGET].values.reshape(-1, 1)
regime_all = df["Regime"].values.reshape(-1, 1).astype(float)

X_train, X_test = X_all[:n_train], X_all[n_train:]
y_train, y_test = y_all[:n_train], y_all[n_train:]
regime_train, regime_test = regime_all[:n_train], regime_all[n_train:]
days_test = df["Day"].values[n_train:]

results = []
predictions = {}


def evaluate(name, y_true, y_pred, train_time):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    results.append({"metod": name, "RMSE": rmse, "MAE": mae, "R2": r2, "vreme_s": train_time})
    predictions[name] = y_pred
    print(f"{name:42s}  RMSE={rmse:8.5f}  MAE={mae:8.5f}  R2={r2:7.3f}  t={train_time:6.2f}s")


# xanfis i gilardi rade sa rucno skaliranim podacima (sklearn StandardScaler)
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)
Xs_train, Xs_test = x_scaler.transform(X_train), x_scaler.transform(X_test)
ys_train, ys_test = y_scaler.transform(y_train).ravel(), y_scaler.transform(y_test).ravel()


def inv_y(y_scaled):
    return y_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()


print("=" * 95)
print("1-3) xanfis: classic hybrid / gradient-only / bio-inspired (PSO)")
print("=" * 95)
from xanfis import AnfisRegressor, GdAnfisRegressor, BioAnfisRegressor

t0 = time.time()
m1 = AnfisRegressor(num_rules=6, mf_class="Gaussian", epochs=100, batch_size=16,
                     optim="Adam", verbose=False, seed=SEED)
m1.fit(Xs_train, ys_train)
t1 = time.time() - t0
evaluate("xanfis - classic hybrid (Adam)", y_test, inv_y(m1.predict(Xs_test)), t1)

t0 = time.time()
m2 = GdAnfisRegressor(num_rules=6, mf_class="Gaussian", epochs=100, batch_size=16,
                       optim="RMSprop", verbose=False, seed=SEED)
m2.fit(Xs_train, ys_train)
t2 = time.time() - t0
evaluate("xanfis - gradient-only (RMSprop)", y_test, inv_y(m2.predict(Xs_test)), t2)

t0 = time.time()
m3 = BioAnfisRegressor(num_rules=6, mf_class="Gaussian", optim="OriginalPSO",
                        optim_params={"epoch": 40, "pop_size": 20}, obj_name="MSE",
                        verbose=False, seed=SEED)
m3.fit(Xs_train, ys_train)
t3 = time.time() - t0
evaluate("xanfis - bio-inspired (PSO)", y_test, inv_y(m3.predict(Xs_test)), t3)

print("=" * 95)
print("4) gabrielegilardi/ANFIS (samostalna PSO implementacija)")
print("=" * 95)
import sys
sys.path.insert(0, str(BASE_DIR / "ANFIS" / "Code_Python"))
import anfis as anf
import utils as utl
import pso as pso_mod

n_mf = [2, 2, 2, 2]
n_outputs = 1
nPop, epochs_gg = 30, 120

Xn_tr, norm_param = utl.normalize_data(X_train)
Xn_te = utl.normalize_data(X_test, norm_param)
LB, UB = utl.bounds_pso(Xn_tr, n_mf, n_outputs)
Y_tr_scaled, scal_param = utl.scale_data(y_train)


def inv_y_gg(Ys):
    X_min, X_max = scal_param
    return (0.5 * (np.asarray(Ys).ravel() + 1.0) * (X_max - X_min) + X_min)


learners = [anf.ANFIS(n_mf=n_mf, n_outputs=n_outputs) for _ in range(nPop)]


def interface_pso(theta, args):
    Xn, Yn, lrns = args
    J = np.zeros(theta.shape[0])
    for i in range(theta.shape[0]):
        J[i] = lrns[i].create_model(theta[i, :], (Xn, Yn))
    return J


t0 = time.time()
np.random.seed(SEED)
theta, info = pso_mod.PSO(interface_pso, LB, UB, nPop=nPop, epochs=epochs_gg,
                           args=(Xn_tr, Y_tr_scaled, learners))
t4 = time.time() - t0
best_learner = learners[info[1]]
y_pred_gg = inv_y_gg(best_learner.eval_data(Xn_te))
evaluate("gabrielegilardi/ANFIS (PSO)", y_test, y_pred_gg, t4)

print("=" * 95)
print("5-6) sanfis: plain rezim i S-ANFIS (regime-switching) rezim")
print("=" * 95)
import torch
import torch.nn as nn
from sanfis import SANFIS

torch.manual_seed(SEED)

# hronoloski izdvojen validacioni deo iz trening skupa (sanfis.fit() zahteva valid_data)
n_valid = int(n_train * 0.15)
n_fit = n_train - n_valid

Xt_fit = torch.tensor(X_train[:n_fit], dtype=torch.float32)
Xt_val = torch.tensor(X_train[n_fit:], dtype=torch.float32)
Xt_test = torch.tensor(X_test, dtype=torch.float32)
yt_fit = torch.tensor(y_train[:n_fit], dtype=torch.float32)
yt_val = torch.tensor(y_train[n_fit:], dtype=torch.float32)
yt_test = torch.tensor(y_test, dtype=torch.float32)
# S-ANFIS rule layer u ovoj implementaciji zahteva najmanje 2 "state"/premisa
# promenljive, pa se premisa prostor sastoji od Regime (predznak Input_1,
# binarni prekidac rezima) i samog Input_1 (kontinualna jacina vodeceg trzista)
state_all = np.column_stack([regime_all.ravel(), X_all[:, 0]])
state_train, state_test = state_all[:n_train], state_all[n_train:]

Rt_fit = torch.tensor(state_train[:n_fit], dtype=torch.float32)
Rt_val = torch.tensor(state_train[n_fit:], dtype=torch.float32)
Rt_test = torch.tensor(state_test, dtype=torch.float32)

membfuncs_plain = [
    {"function": "gaussian", "n_memb": 2,
     "params": {"mu": {"value": [-1.0, 1.0], "trainable": True},
                "sigma": {"value": [1.0, 1.0], "trainable": True}}}
    for _ in range(len(ORDINARY_FEATURES))
]

t0 = time.time()
model_plain = SANFIS(membfuncs_plain, n_input=len(ORDINARY_FEATURES), scale="Std")
optimizer = torch.optim.Adam(model_plain.parameters(), lr=0.01)
model_plain.fit([Xt_fit, yt_fit], [Xt_val, yt_val], optimizer, nn.MSELoss(),
                batch_size=16, epochs=150, patience=20, disable_output=True)
t5 = time.time() - t0
y_pred_plain = model_plain.predict(Xt_test).numpy().ravel()
evaluate("sanfis - plain (bez state razdvajanja)", y_test, y_pred_plain, t5)

# S-ANFIS rezim: state = [Regime, Input_1], posledica = ordinary features
membfuncs_state = [
    {"function": "gaussian", "n_memb": 2,
     "params": {"mu": {"value": [-1.0, 1.0], "trainable": True},
                "sigma": {"value": [1.0, 1.0], "trainable": True}}},
    {"function": "gaussian", "n_memb": 2,
     "params": {"mu": {"value": [-1.0, 1.0], "trainable": True},
                "sigma": {"value": [1.0, 1.0], "trainable": True}}},
]

t0 = time.time()
torch.manual_seed(SEED)
model_sanfis = SANFIS(membfuncs_state, n_input=len(ORDINARY_FEATURES), scale="Std")
optimizer2 = torch.optim.Adam(model_sanfis.parameters(), lr=0.01)
model_sanfis.fit([Rt_fit, Xt_fit, yt_fit], [Rt_val, Xt_val, yt_val], optimizer2, nn.MSELoss(),
                 batch_size=16, epochs=150, patience=20, disable_output=True)
t6 = time.time() - t0
y_pred_sanfis = model_sanfis.predict([Rt_test, Xt_test]).numpy().ravel()
evaluate("sanfis - S-ANFIS (Regime kao state)", y_test, y_pred_sanfis, t6)

res_df = pd.DataFrame(results).sort_values("RMSE")
res_df.to_csv(BASE_DIR / "results.csv", index=False)
print("\n" + "=" * 95)
print("REZULTATI (sortirano po RMSE):")
print(res_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
res_sorted = res_df.sort_values("RMSE")
bars = ax.barh(res_sorted["metod"], res_sorted["RMSE"], color="#4C72B0")
ax.set_xlabel("RMSE (ISE dnevni prinos)")
ax.set_title("Poredjenje metoda treniranja ANFIS-a - predvidjanje ISE prinosa")
ax.invert_yaxis()
for bar, val in zip(bars, res_sorted["RMSE"]):
    ax.text(val, bar.get_y() + bar.get_height() / 2, f" {val:.5f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(BASE_DIR / "rmse_comparison.png", dpi=150)
plt.close()

best_name = res_sorted.iloc[0]["metod"]
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(days_test, y_test.ravel(), label="Stvarni prinos", color="black", linewidth=1.3)
ax.plot(days_test, predictions[best_name], label=f"Predikcija ({best_name})", color="#DD8452", linewidth=1.3)
ax.set_xlabel("Dan (test skup)")
ax.set_ylabel("ISE dnevni prinos")
ax.set_title(f"Predikcija vs. stvarna vrednost - najbolji model: {best_name}")
ax.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "best_model_prediction.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(days_test, y_test.ravel(), label="Stvarni prinos", color="black", linewidth=1.3)
ax.plot(days_test, predictions["sanfis - plain (bez state razdvajanja)"], label="sanfis - plain", color="#55A868", linewidth=1.1, alpha=0.85)
ax.plot(days_test, predictions["sanfis - S-ANFIS (Regime kao state)"], label="sanfis - S-ANFIS (regime)", color="#C44E52", linewidth=1.1, alpha=0.85)
ax.set_xlabel("Dan (test skup)")
ax.set_ylabel("ISE dnevni prinos")
ax.set_title("Uticaj razdvajanja premisa/posledica: plain vs. S-ANFIS (regime-aware)")
ax.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "sanfis_vs_plain.png", dpi=150)
plt.close()

print("\nGrafici sacuvani: rmse_comparison.png, best_model_prediction.png, sanfis_vs_plain.png")
print("Rezultati sacuvani: results.csv")
