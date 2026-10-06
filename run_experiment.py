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
  1) referentne (baseline) metode: naivna prognoza (predvidjanje 0) i
     linearna regresija - u dve varijante, sa i bez Output_2 kolone
     (provera osetljivosti rezultata na ukljucivanje te kolone)
  2) 10 slucajnih seedova za svaki stohasticki model
  3) upareni Wilcoxonov test rangova sa znakom nad kvadratnim greskama
     po danu (n=108), za kljucne parove modela
  4) klasicki hibridni trener sa 300 epoha - provera
     hipoteze da je slabiji rezultat hibridnog modela posledica
     podtreniranosti sa 100 epoha

"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

# hronoloski 80/20 split
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

# xanfis i gilardi rade sa rucno skaliranim podacima (sklearn StandardScaler)
# skaliranje iskljucivo na treningu, test se transformise istim parametrima
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)
Xs_train, Xs_test = x_scaler.transform(X_train), x_scaler.transform(X_test)
ys_train, ys_test = y_scaler.transform(y_train).ravel(), y_scaler.transform(y_test).ravel()


def inv_y(y_scaled):
    """model radi nad skaliranim y, pa predikcije vracamo u skalu prinosa"""
    return y_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()


# skup seedova za ponovljena pokretanja stohastickih modela
SEEDS = [42, 7, 123, 2024, 555, 91, 3, 999, 17, 314]

# per_model[name] = {"rmse": [...], "mae": [...], "r2": [...], "t": [...], "pred": {seed: y_pred}}
per_model = {}

def record(name, seed, y_true, y_pred, train_time):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    e = per_model.setdefault(name, {"rmse": [], "mae": [], "r2": [], "t": [], "pred": {}})
    e["rmse"].append(np.sqrt(mean_squared_error(y_true, y_pred)))
    e["mae"].append(mean_absolute_error(y_true, y_pred))
    e["r2"].append(r2_score(y_true, y_pred))
    e["t"].append(train_time)
    e["pred"][seed] = y_pred


print("=" * 95)
print("0) REFERENTNE (BASELINE) METODE - deterministicke, jedno pokretanje")
print("=" * 95)
baselines = {}

# naivna prognoza: konstantna predikcija nule (prosek serije) - donja granica
t0 = time.time()
y_pred_naive = np.zeros(len(y_test))
record("naivna (predikcija 0)", "det", y_test, y_pred_naive, time.time() - t0)
baselines["naivna (predikcija 0)"] = y_pred_naive

# linearna regresija - gornja granica jednostavnih modela, dve varijante
for nm, feats in [("linearna regresija (sa Output_2)", ORDINARY_FEATURES),
                  ("linearna regresija (bez Output_2)", ["Input_1", "Input_2", "Input_3"])]:
    Xa = df[feats].values
    sc = StandardScaler().fit(Xa[:n_train])
    t0 = time.time()
    mlin = LinearRegression().fit(sc.transform(Xa[:n_train]), y_train.ravel())
    yp = mlin.predict(sc.transform(Xa[n_train:]))
    record(nm, "det", y_test, yp, time.time() - t0)
    baselines[nm] = yp
    print(f"{nm:45s}  RMSE={np.sqrt(mean_squared_error(y_test, yp)):.5f}  "
          f"R2={r2_score(y_test, yp):.3f}")

print("=" * 95)
print("1-4) STOHASTICKI MODELI - 10 seedova svaki")
print("=" * 95)

from xanfis import AnfisRegressor, GdAnfisRegressor, BioAnfisRegressor

def run_xanfis_set(seed):
    np.random.seed(seed)

    # 1) klasicki hibridni (gradijent + LSE), 100 epoha
    t0 = time.time()
    m1 = AnfisRegressor(num_rules=6, mf_class="Gaussian", epochs=100, batch_size=16,
                        optim="Adam", verbose=False, seed=seed)
    m1.fit(Xs_train, ys_train)
    record("xanfis - classic hybrid (Adam, 100 epoha)", seed, y_test,
           inv_y(m1.predict(Xs_test)), time.time() - t0)

    # 1b) isti hibridni trener, ali sa 300 epoha - provera hipoteze o podtreniranosti
    t0 = time.time()
    m1b = AnfisRegressor(num_rules=6, mf_class="Gaussian", epochs=300, batch_size=16,
                         optim="Adam", early_stopping=False, verbose=False, seed=seed)
    m1b.fit(Xs_train, ys_train)
    record("xanfis - classic hybrid (Adam, 300 epoha)", seed, y_test,
           inv_y(m1b.predict(Xs_test)), time.time() - t0)

    # 2) iskljucivo gradijentni trener (RMSprop), bez hibridne komponente
    t0 = time.time()
    m2 = GdAnfisRegressor(num_rules=6, mf_class="Gaussian", epochs=100, batch_size=16,
                          optim="RMSprop", verbose=False, seed=seed)
    m2.fit(Xs_train, ys_train)
    record("xanfis - gradient-only (RMSprop)", seed, y_test,
           inv_y(m2.predict(Xs_test)), time.time() - t0)

    # 3) bio-inspirisana varijanta: PSO trazi parametre, konsekvence i dalje LSE
    t0 = time.time()
    m3 = BioAnfisRegressor(num_rules=6, mf_class="Gaussian", optim="OriginalPSO",
                           optim_params={"epoch": 40, "pop_size": 20}, obj_name="MSE",
                           verbose=False, seed=seed)
    m3.fit(Xs_train, ys_train)
    record("xanfis - bio-inspired (PSO)", seed, y_test,
           inv_y(m3.predict(Xs_test)), time.time() - t0)

import sys
sys.path.insert(0, str(BASE_DIR / "ANFIS" / "Code_Python"))
import anfis as anf
import utils as utl
import pso as pso_mod

def run_gilardi(seed):
    np.random.seed(seed)
    n_mf = [2, 2, 2, 2]
    n_outputs = 1
    nPop, epochs_gg = 30, 120
    Xn_tr, norm_param = utl.normalize_data(X_train)
    Xn_te = utl.normalize_data(X_test, norm_param)
    LB, UB = utl.bounds_pso(Xn_tr, n_mf, n_outputs)
    Y_tr_scaled, scal_param = utl.scale_data(y_train)
    X_min, X_max = scal_param
    learners = [anf.ANFIS(n_mf=n_mf, n_outputs=n_outputs) for _ in range(nPop)]

    def interface_pso(theta, args):
        Xn, Yn, lrns = args
        J = np.zeros(theta.shape[0])
        for i in range(theta.shape[0]):
            J[i] = lrns[i].create_model(theta[i, :], (Xn, Yn))
        return J

    t0 = time.time()
    theta, info = pso_mod.PSO(interface_pso, LB, UB, nPop=nPop, epochs=epochs_gg,
                              args=(Xn_tr, Y_tr_scaled, learners))
    t = time.time() - t0
    best_learner = learners[info[1]]
    yp = best_learner.eval_data(Xn_te)
    yp = 0.5 * (np.asarray(yp).ravel() + 1.0) * (X_max - X_min) + X_min
    record("gabrielegilardi/ANFIS (PSO)", seed, y_test, yp, t)

import torch
import torch.nn as nn
from sanfis import SANFIS

torch.use_deterministic_algorithms(False)

n_valid = int(n_train * 0.15)
n_fit = n_train - n_valid

Xt_fit = torch.tensor(X_train[:n_fit], dtype=torch.float32)
Xt_val = torch.tensor(X_train[n_fit:], dtype=torch.float32)
Xt_test = torch.tensor(X_test, dtype=torch.float32)
yt_fit = torch.tensor(y_train[:n_fit], dtype=torch.float32)
yt_val = torch.tensor(y_train[n_fit:], dtype=torch.float32)
yt_test_t = torch.tensor(y_test, dtype=torch.float32)

# S-ANFIS premise prostor: Regime (binarni prekidac rezima) + Input_1 (jacina vodeceg trzista)
state_all = np.column_stack([regime_all.ravel(), X_all[:, 0]])
state_train = state_all[:n_train]

Rt_fit = torch.tensor(state_train[:n_fit], dtype=torch.float32)
Rt_val = torch.tensor(state_train[n_fit:], dtype=torch.float32)
Rt_test = torch.tensor(state_all[n_train:], dtype=torch.float32)

membfuncs_plain = [
    {"function": "gaussian", "n_memb": 2,
     "params": {"mu": {"value": [-1.0, 1.0], "trainable": True},
                "sigma": {"value": [1.0, 1.0], "trainable": True}}}
    for _ in range(len(ORDINARY_FEATURES))
]
membfuncs_state = [
    {"function": "gaussian", "n_memb": 2,
     "params": {"mu": {"value": [-1.0, 1.0], "trainable": True},
                "sigma": {"value": [1.0, 1.0], "trainable": True}}},
    {"function": "gaussian", "n_memb": 2,
     "params": {"mu": {"value": [-1.0, 1.0], "trainable": True},
                "sigma": {"value": [1.0, 1.0], "trainable": True}}},
]

def run_sanfis(seed):
    torch.manual_seed(seed)

    # plain SANFIS - bez eksplicitnog razdvajanja po rezimu
    t0 = time.time()
    model_plain = SANFIS(membfuncs_plain, n_input=len(ORDINARY_FEATURES), scale="Std")
    optimizer = torch.optim.Adam(model_plain.parameters(), lr=0.01)
    model_plain.fit([Xt_fit, yt_fit], [Xt_val, yt_val], optimizer, nn.MSELoss(),
                    batch_size=16, epochs=150, patience=20, disable_output=True)
    t = time.time() - t0
    record("sanfis - plain (bez state razdvajanja)", seed, y_test,
           model_plain.predict(Xt_test).numpy().ravel(), t)

    torch.manual_seed(seed)
    # S-ANFIS - model eksplicitno dobija informaciju o rezimu trzista u premisi
    t0 = time.time()
    model_sanfis = SANFIS(membfuncs_state, n_input=len(ORDINARY_FEATURES), scale="Std")
    optimizer2 = torch.optim.Adam(model_sanfis.parameters(), lr=0.01)
    model_sanfis.fit([Rt_fit, Xt_fit, yt_fit], [Rt_val, Xt_val, yt_val], optimizer2,
                     nn.MSELoss(), batch_size=16, epochs=150, patience=20,
                     disable_output=True)
    t = time.time() - t0
    record("sanfis - S-ANFIS (Regime kao state)", seed, y_test,
           model_sanfis.predict([Rt_test, Xt_test]).numpy().ravel(), t)


for si, s in enumerate(SEEDS):
    run_xanfis_set(s)
    run_gilardi(s)
    run_sanfis(s)
    done = sorted(per_model.keys())
    best_so_far = min(done, key=lambda k: np.mean(per_model[k]["rmse"]))
    print(f"seed {s:5d} | pokretanje {si+1}/{len(SEEDS)} | "
          f"trenutno najbolji (mean RMSE): {best_so_far}")

# rezultati
rows = []
for name, e in per_model.items():
    rows.append({
        "metod": name,
        "RMSE_mean": np.mean(e["rmse"]), "RMSE_std": np.std(e["rmse"]),
        "MAE_mean": np.mean(e["mae"]), "MAE_std": np.std(e["mae"]),
        "R2_mean": np.mean(e["r2"]), "R2_std": np.std(e["r2"]),
        "vreme_mean": np.mean(e["t"]), "vreme_std": np.std(e["t"]),
        "n_pokretanja": len(e["rmse"]),
    })
res_df = pd.DataFrame(rows).sort_values("RMSE_mean")
res_df.to_csv(BASE_DIR / "results.csv", index=False)

print("\n" + "=" * 95)
print("REZULTATI - prosireni protokol (mean +- std po seedovima):")
print("=" * 95)
for _, r in res_df.iterrows():
    print(f"{r['metod']:45s} RMSE={r['RMSE_mean']:.5f}+-{r['RMSE_std']:.5f}  "
          f"MAE={r['MAE_mean']:.5f}+-{r['MAE_std']:.5f}  R2={r['R2_mean']:.3f}+-{r['R2_std']:.3f}  "
          f"t={r['vreme_mean']:.2f}s")

# reprezentativni run = pokretanje sa medijanom RMSE (za grafike i uparene testove)
def median_run(name):
    e = per_model[name]
    s = sorted(e["rmse"])[len(e["rmse"]) // 2]
    seed = list(e["pred"].keys())[e["rmse"].index(s)]
    return e["pred"][seed], seed

# wilcoxon
PAIRS = [
    ("xanfis - gradient-only (RMSprop)", "linearna regresija (sa Output_2)"),
    ("sanfis - S-ANFIS (Regime kao state)", "sanfis - plain (bez state razdvajanja)"),
    ("xanfis - classic hybrid (Adam, 300 epoha)", "xanfis - classic hybrid (Adam, 100 epoha)"),
    ("xanfis - bio-inspired (PSO)", "gabrielegilardi/ANFIS (PSO)"),
]
wilcox_rows = []
print("\n" + "=" * 95)
print("UPARENI WILCOXONOV TEST (kvadratne greske po danu, reprezentativni runovi):")
print("=" * 95)
for a, b in PAIRS:
    ya, _ = median_run(a)
    if b in per_model:
        yb, _ = median_run(b)
    else:
        yb = baselines[b]
    ea, eb = (y_test.ravel() - ya) ** 2, (y_test.ravel() - yb) ** 2
    try:
        stat, p = wilcoxon(ea, eb)
    except ValueError:
        stat, p = 0.0, 1.0
    wilcox_rows.append({"poredjenje": f"{a}  VS  {b}", "W": stat, "p_vrednost": p})
    print(f"{a}\n  VS  {b}\n  W={stat:.1f}  p={p:.4f}  {'ZNACAJNO' if p < 0.05 else 'n.z.'}\n")
pd.DataFrame(wilcox_rows).to_csv(BASE_DIR / "wilcoxon.csv", index=False)

# grafici
fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.barh(res_df["metod"], res_df["RMSE_mean"], xerr=res_df["RMSE_std"],
               color="#4C72B0", error_kw=dict(lw=1, capsize=3))
ax.set_xlabel("RMSE (ISE dnevni prinos), mean +- std po 10 seedova")
ax.set_title("Poredjenje metoda treniranja ANFIS-a - prosireni protokol")
ax.invert_yaxis()
for bar, val in zip(bars, res_df["RMSE_mean"]):
    ax.text(val, bar.get_y() + bar.get_height() / 2, f" {val:.5f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(BASE_DIR / "rmse_comparison.png", dpi=150)
plt.close()

best_name = res_df.iloc[0]["metod"]
y_best, best_seed = median_run(best_name)
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(days_test, y_test.ravel(), label="Stvarni prinos", color="black", linewidth=1.3)
ax.plot(days_test, y_best, label=f"Predikcija ({best_name}, seed={best_seed})",
        color="#DD8452", linewidth=1.3)
ax.set_xlabel("Dan (test skup)")
ax.set_ylabel("ISE dnevni prinos")
ax.set_title(f"Predikcija vs. stvarna vrednost - najbolji model (reprezentativni run)")
ax.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "best_model_prediction.png", dpi=150)
plt.close()

y_plain, _ = median_run("sanfis - plain (bez state razdvajanja)")
y_sf, _ = median_run("sanfis - S-ANFIS (Regime kao state)")
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(days_test, y_test.ravel(), label="Stvarni prinos", color="black", linewidth=1.3)
ax.plot(days_test, y_plain, label="sanfis - plain", color="#55A868", linewidth=1.1, alpha=0.85)
ax.plot(days_test, y_sf, label="sanfis - S-ANFIS (regime)", color="#C44E52", linewidth=1.1, alpha=0.85)
ax.set_xlabel("Dan (test skup)")
ax.set_ylabel("ISE dnevni prinos")
ax.set_title("Uticaj razdvajanja premisa/posledica: plain vs. S-ANFIS (reprezentativni runovi)")
ax.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "sanfis_vs_plain.png", dpi=150)
plt.close()

print("Sacuvano: results.csv, wilcoxon.csv, rmse_comparison.png,")
print("           best_model_prediction.png, sanfis_vs_plain.png")
