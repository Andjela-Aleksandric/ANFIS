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

SEEDS = [42, 7, 123, 2024, 555, 91, 3, 999, 17, 314]
# Regularizacija LSE (posledicnog) koraka kod klasicnog hibridnog trenera -
# sprecava numericku nestabilnost kad se trening pusti na 300 epoha bez
# early stoppinga
HYBRID_REG_LAMBDA = 1.0
# Kandidati za broj ANFIS pravila - biramo izmedju njih preko validacije,
# ne nagadjanjem (vidi hyperparameter tuning blok nize). Drzimo ih na 4/6 -
# 8 ili 10 pravila na 8 ulaza je destabilisalo PSO varijante (prevelik
# prostor pretrage za isti budzet optimizacije)
RULE_CANDIDATES = [4, 6]
# Koliko dana unazad (lag) koristimo kao ulaze - vise lagova znaci da model
# vidi duzu istoriju (ne samo juce, nego i prekjuce), sto moze da uhvati
# eventualni momentum/autokorelaciju koju jedan lag propusta. Ograniceno na
# 2 laga (= 8 ulaza sa 4 promenljive) jer SANFIS biblioteka tvrdo podrzava
# najvise 8 ulaznih promenljivih - sa 3 laga (12 ulaza) pukne.
LAG_STEPS = [1, 2]

# Ucitavamo dataset i pripremamo ulaze koje cemo koristiti u modelu.
# Ulazi (Input_1-3, Output_2) se pomeraju za LAG_STEPS dana unazad u odnosu
# na cilj (Output_1) - predvidjamo danasnji prinos na osnovu proslih
# vrednosti, sto je prava jednodnevna prognoza. Output_2 i Output_1 su,
# prema dokumentaciji dataset-a, isti ISE prinos izrazen u dve razlicite
# valute ISTOG dana - kad bi se Output_2 koristio kao ulaz istog dana,
# model bi prakticno "video" odgovor pre predikcije (curenje informacija),
# zato se i on pomera unazad kao i ostali ulazi.
df_raw = pd.read_csv(BASE_DIR / "ise_features.csv")
RAW_FEATURES = ["Input_1", "Input_2", "Input_3", "Output_2"]
TARGET = "Output_1"

df = df_raw.copy()
for col in RAW_FEATURES:
    for lag in LAG_STEPS:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
df["Regime"] = (df["Input_1"].shift(1) > 0).astype(int)
df = df.dropna().reset_index(drop=True)
ORDINARY_FEATURES = [f"{c}_lag{lag}" for c in RAW_FEATURES for lag in LAG_STEPS]
df["Day"] = np.arange(len(df))
df.to_csv(BASE_DIR / "ise_features_prepared.csv", index=False)

print("=" * 90)
print(f"N={len(df)} | features={ORDINARY_FEATURES} | target={TARGET}")
print("=" * 90)

# Kod vremenske serije ne mesamo podatke, vec poslednjih 20% ostavljamo za test
n = len(df)
n_train = int(0.8 * n)

X_all = df[ORDINARY_FEATURES].values
y_all = df[TARGET].values.reshape(-1, 1)
regime_all = df["Regime"].values.reshape(-1, 1).astype(float)

X_train, X_test = X_all[:n_train], X_all[n_train:]
y_train, y_test = y_all[:n_train], y_all[n_train:]
regime_train, regime_test = regime_all[:n_train], regime_all[n_train:]
days_test = df["Day"].values[n_train:]

# Test skup ne sme da utice na parametre skaliranja
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)
Xs_train = x_scaler.transform(X_train)
Xs_test = x_scaler.transform(X_test)
ys_train = y_scaler.transform(y_train).ravel()


def inv_y(y_scaled):
    return y_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()

# Validacioni deo - poslednjih 15% TRENING perioda (hronoloski). Koristi se
# i za biranje broja ANFIS pravila nize, i kasnije za SANFIS early stopping.
# Test skup se u ovom koraku nigde ne dodiruje.
n_valid = int(n_train * 0.15)
n_fit = n_train - n_valid

# Testiramo tri nacina treniranja iz X-ANFIS biblioteke
from xanfis import AnfisRegressor, GdAnfisRegressor, BioAnfisRegressor

# Biranje broja ANFIS pravila (num_rules) preko validacije - probamo par
# kandidata, treniramo kratko (jedan seed, 100 epoha) na "tuning-trening"
# delu i merimo gresku na validacionom delu, biramo najbolji. Ovo je
# legitimno podesavanje hiperparametra (test skup se ne koristi nigde u
# ovom koraku), za razliku od pogadjanja broja pravila unapred.
Xs_tune_fit, Xs_tune_val = Xs_train[:n_fit], Xs_train[n_fit:]
ys_tune_fit = ys_train[:n_fit]
y_tune_val = y_train[n_fit:].ravel()

print("Biranje broja ANFIS pravila (num_rules) preko validacionog skupa:")
best_rules, best_val_rmse = RULE_CANDIDATES[0], np.inf
for rules in RULE_CANDIDATES:
    m = AnfisRegressor(num_rules=rules, mf_class="Gaussian", epochs=100,
                       batch_size=16, optim="Adam", reg_lambda=HYBRID_REG_LAMBDA,
                       verbose=False, seed=42)
    m.fit(Xs_tune_fit, ys_tune_fit)
    val_pred = inv_y(m.predict(Xs_tune_val))
    val_rmse = np.sqrt(mean_squared_error(y_tune_val, val_pred))
    print(f"  num_rules={rules}: validacioni RMSE={val_rmse:.6f}")
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_rules = rules

XANFIS_RULES = best_rules
print(f"Izabrano: num_rules={XANFIS_RULES} (validacioni RMSE={best_val_rmse:.6f})")
print("=" * 90)

# Ovde cuvamo rezultate svakog pokretanja da bismo kasnije mogli da ih uporedimo
per_model = {}
baselines = {}


def record(name, seed, y_true, y_pred, train_time):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    e = per_model.setdefault(name, {"rmse": [], "mae": [], "r2": [], "t": [], "pred": {}})
    e["rmse"].append(np.sqrt(mean_squared_error(y_true, y_pred)))
    e["mae"].append(mean_absolute_error(y_true, y_pred))
    e["r2"].append(r2_score(y_true, y_pred))
    e["t"].append(train_time)
    e["pred"][seed] = y_pred

# Pre ANFIS modela pravimo jednostavne baseline modele za poredjenje
t0 = time.time()
naive = np.zeros(len(y_test))
record("naivna (predikcija 0)", "det", y_test, naive, time.time() - t0)
baselines["naivna (predikcija 0)"] = naive

t0 = time.time()
lin = LinearRegression().fit(Xs_train, y_train.ravel())
y_lin = lin.predict(Xs_test)
record("linearna regresija", "det", y_test, y_lin, time.time() - t0)
baselines["linearna regresija"] = y_lin


def run_xanfis(seed):
    np.random.seed(seed)

    # reg_lambda dodat i ovde (100 epoha) radi doslednosti - u suprotnom bi
    # samo 300-epoha varijanta bila regularizovana, sto bi iskrivilo
    # poredjenje "da li duze treniranje pomaze"
    t0 = time.time()
    m = AnfisRegressor(num_rules=XANFIS_RULES, mf_class="Gaussian", epochs=100,
                       batch_size=16, optim="Adam", reg_lambda=HYBRID_REG_LAMBDA,
                       verbose=False, seed=seed)
    m.fit(Xs_train, ys_train)
    record("xanfis - classic hybrid (Adam, 100)", seed, y_test,
           inv_y(m.predict(Xs_test)), time.time() - t0)

    t0 = time.time()
    m = AnfisRegressor(num_rules=XANFIS_RULES, mf_class="Gaussian", epochs=300,
                       batch_size=16, optim="Adam", early_stopping=False,
                       reg_lambda=HYBRID_REG_LAMBDA,
                       verbose=False, seed=seed)
    m.fit(Xs_train, ys_train)
    record("xanfis - classic hybrid (Adam, 300)", seed, y_test,
           inv_y(m.predict(Xs_test)), time.time() - t0)

    t0 = time.time()
    m = GdAnfisRegressor(num_rules=XANFIS_RULES, mf_class="Gaussian", epochs=100,
                         batch_size=16, optim="RMSprop", verbose=False, seed=seed)
    m.fit(Xs_train, ys_train)
    record("xanfis - gradient-only (RMSprop)", seed, y_test,
           inv_y(m.predict(Xs_test)), time.time() - t0)

    # malo veci PSO budzet nego ranije - sad je 8 ulaza umesto 4, roju treba
    # vise prostora da ne zaglavi u losem resenju
    t0 = time.time()
    m = BioAnfisRegressor(num_rules=XANFIS_RULES, mf_class="Gaussian",
                          optim="OriginalPSO", optim_params={"epoch": 60, "pop_size": 30},
                          obj_name="MSE", verbose=False, seed=seed)
    m.fit(Xs_train, ys_train)
    record("xanfis - bio-inspired (PSO)", seed, y_test,
           inv_y(m.predict(Xs_test)), time.time() - t0)

# Ovaj model koristi originalnu Gilardi implementaciju sa PSO optimizacijom
import sys
sys.path.insert(0, str(BASE_DIR / "ANFIS" / "Code_Python"))
import anfis as anf
import utils as utl
import pso as pso_mod


def run_gilardi(seed):
    np.random.seed(seed)
    n_mf = [2] * len(ORDINARY_FEATURES)
    n_outputs = 1
    nPop, epochs = 40, 150  # isto, vise ulaza trazi veci budzet

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
    theta, info = pso_mod.PSO(interface_pso, LB, UB, nPop=nPop,
                               epochs=epochs, args=(Xn_tr, Y_tr_scaled, learners))
    best = learners[info[1]]
    yp = best.eval_data(Xn_te)
    yp = 0.5 * (np.asarray(yp).ravel() + 1.0) * (X_max - X_min) + X_min
    record("gabrielegilardi/ANFIS (PSO)", seed, y_test, yp, time.time() - t0)

# Ovde poredimo obican SANFIS sa S-ANFIS pristupom
import torch
import torch.nn as nn
from sanfis import SANFIS

torch.use_deterministic_algorithms(False)

Xt_fit = torch.tensor(X_train[:n_fit], dtype=torch.float32)
Xt_val = torch.tensor(X_train[n_fit:], dtype=torch.float32)
Xt_test = torch.tensor(X_test, dtype=torch.float32)
yt_fit = torch.tensor(y_train[:n_fit], dtype=torch.float32)
yt_val = torch.tensor(y_train[n_fit:], dtype=torch.float32)

membfuncs_plain = [
    {"function": "gaussian", "n_memb": 2,
     "params": {"mu": {"value": [-1.0, 1.0], "trainable": True},
                "sigma": {"value": [1.0, 1.0], "trainable": True}}}
    for _ in ORDINARY_FEATURES
]

# S-ANFIS koristi poseban state prostor koji opisuje trenutni rezim sistema -
# i state koristimo iz prethodnog dana, isti princip kao i ostali ulazi
state_signal = df["Input_1_lag1"].values
state_all = np.column_stack([regime_all.ravel(), state_signal])
state_train = state_all[:n_train]

membfuncs_state = [
    {"function": "gaussian", "n_memb": 2,
     "params": {"mu": {"value": [-1.0, 1.0], "trainable": True},
                "sigma": {"value": [1.0, 1.0], "trainable": True}}},
    {"function": "gaussian", "n_memb": 2,
     "params": {"mu": {"value": [-1.0, 1.0], "trainable": True},
                "sigma": {"value": [1.0, 1.0], "trainable": True}}}
]


def run_sanfis(seed):
    torch.manual_seed(seed)

    # Kod plain ANFIS-a svi ulazi se tretiraju kao obicni ulazi modela
    model = SANFIS(membfuncs_plain, n_input=len(ORDINARY_FEATURES), scale="Std")
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    t0 = time.time()
    model.fit([Xt_fit, yt_fit], [Xt_val, yt_val], opt, nn.MSELoss(),
              batch_size=16, epochs=150, patience=20, disable_output=True)
    record("sanfis - plain", seed, y_test,
           model.predict(Xt_test).numpy().ravel(), time.time() - t0)

    # Kod S-ANFIS-a odvajamo informacije koje opisuju stanje od ulaza koji direktno vode ka izlazu
    torch.manual_seed(seed)
    model = SANFIS(membfuncs_state, n_input=len(ORDINARY_FEATURES), scale="Std")
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    Rt_fit = torch.tensor(state_train[:n_fit], dtype=torch.float32)
    Rt_val = torch.tensor(state_train[n_fit:], dtype=torch.float32)
    Rt_test = torch.tensor(state_all[n_train:], dtype=torch.float32)
    t0 = time.time()
    model.fit([Rt_fit, Xt_fit, yt_fit], [Rt_val, Xt_val, yt_val], opt, nn.MSELoss(),
              batch_size=16, epochs=150, patience=20, disable_output=True)
    record("sanfis - S-ANFIS", seed, y_test,
           model.predict([Rt_test, Xt_test]).numpy().ravel(), time.time() - t0)

# Sve modele pokrecemo kroz isti skup seedova
for i, seed in enumerate(SEEDS, start=1):
    run_xanfis(seed)
    run_gilardi(seed)
    run_sanfis(seed)
    print(f"Seed {seed} | {i}/{len(SEEDS)}")

# Na kraju racunamo prosecne rezultate i standardnu devijaciju
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

# Za statisticko poredjenje uzimamo reprezentativno pokretanje svakog modela
def median_run(name):
    e = per_model[name]
    idx = np.argsort(e["rmse"])[len(e["rmse"]) // 2]
    seed = list(e["pred"].keys())[idx]
    return e["pred"][seed], seed

pairs = [
    ("xanfis - gradient-only (RMSprop)", "linearna regresija"),
    ("sanfis - S-ANFIS", "sanfis - plain"),
    ("xanfis - classic hybrid (Adam, 300)", "xanfis - classic hybrid (Adam, 100)"),
    ("xanfis - bio-inspired (PSO)", "gabrielegilardi/ANFIS (PSO)"),
]

wilcox = []
for a, b in pairs:
    ya, sa = median_run(a)
    if b in per_model:
        yb, sb = median_run(b)
    else:
        yb, sb = baselines[b], "det"
    ea = (y_test.ravel() - ya) ** 2
    eb = (y_test.ravel() - yb) ** 2
    try:
        stat, p = wilcoxon(ea, eb)
    except ValueError:
        stat, p = 0.0, 1.0
    wilcox.append({"poredjenje": f"{a} VS {b}", "seed_a": sa, "seed_b": sb,
                   "W": stat, "p_vrednost": p})

pd.DataFrame(wilcox).to_csv(BASE_DIR / "wilcoxon.csv", index=False)

# Pravljenje nekoliko jednostavnih grafika za kasniju analizu rezultata
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.barh(res_df["metod"], res_df["RMSE_mean"], xerr=res_df["RMSE_std"],
        error_kw=dict(lw=1, capsize=3))
ax.set_xlabel("RMSE, mean ± std")
ax.set_title("ANFIS benchmark")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(BASE_DIR / "rmse_comparison.png", dpi=150)
plt.close()

best_name = res_df.iloc[0]["metod"]
y_best, best_seed = median_run(best_name)
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(days_test, y_test.ravel(), label="Stvarni prinos", linewidth=1.3)
ax.plot(days_test, y_best, label=f"Predikcija: {best_name}, seed={best_seed}", linewidth=1.3)
ax.set_xlabel("Dan test skupa")
ax.set_ylabel("ISE dnevni prinos")
ax.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "best_model_prediction.png", dpi=150)
plt.close()

plain, _ = median_run("sanfis - plain")
sstate, _ = median_run("sanfis - S-ANFIS")
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(days_test, y_test.ravel(), label="Stvarni prinos", linewidth=1.3)
ax.plot(days_test, plain, label="SANFIS plain", linewidth=1.1)
ax.plot(days_test, sstate, label="S-ANFIS", linewidth=1.1)
ax.set_xlabel("Dan test skupa")
ax.set_ylabel("ISE dnevni prinos")
ax.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "sanfis_vs_plain.png", dpi=150)
plt.close()

print("Sacuvano:")
for f in [
    "ise_features_prepared.csv",
    "results.csv",
    "wilcoxon.csv",
    "rmse_comparison.png",
    "best_model_prediction.png",
    "sanfis_vs_plain.png",
]:
    print(" -", f)