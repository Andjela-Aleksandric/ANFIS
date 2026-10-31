import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, accuracy_score,
                              precision_score, recall_score, f1_score)

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "podaci KNFIS (version 1).xlsx"

FEATURES = ["BOL", "ZAMUCENOST", "FEBRILNOST", "LEUKOCITI", "KULTURA"]
TARGET = "BOLEST"
EPOCHS = 100  # isto kao prvi eksperiment
N_MF_PER_INPUT = 2  # isto kao prvi eksperiment

# ucitavanje iz Sheet1 (originalni, kompletni podaci), sveza podela, bez
# ikakvog uklanjanja/filtriranja redova - dataset je gotov i koristi se ceo
xl = pd.ExcelFile(DATA_PATH)
sheet1 = xl.parse("Sheet1")[FEATURES + [TARGET]].reset_index(drop=True)

print("=" * 90)
print(f"Sheet1: {len(sheet1)} pacijenata (ceo skup, bez izbacivanja)")

X_all = sheet1[FEATURES].values.astype(float)
y_all = sheet1[TARGET].values.astype(int)

# drugi random seed nego prvi eksperiment (42), i drugi odnos (70/30 umesto
# ~79/21) - namerno manji trening skup, generalizacija je teza
train_df_idx, test_df_idx = train_test_split(
    np.arange(len(sheet1)), test_size=0.30, stratify=y_all, random_state=2026,
)
X_train_raw, y_train = X_all[train_df_idx], y_all[train_df_idx]
X_test_raw, y_test = X_all[test_df_idx], y_all[test_df_idx]

print(f"Train: {len(X_train_raw)} pacijenata | balans BOLEST: {pd.Series(y_train).value_counts().to_dict()}")
print(f"Test:  {len(X_test_raw)} pacijenata  | balans BOLEST: {pd.Series(y_test).value_counts().to_dict()}")
print("=" * 90)

# modeli - identicno prvom eksperimentu (ista 7 metoda, isti hiperparametri)
from xanfis import AnfisClassifier, GdAnfisClassifier, BioAnfisClassifier


def run_xanfis_hybrid(X_fit, y_fit, X_eval):
    m = AnfisClassifier(epochs=EPOCHS, verbose=False)
    m.fit(X_fit, y_fit)
    return m.predict(X_eval)


def run_xanfis_gradient(X_fit, y_fit, X_eval):
    m = GdAnfisClassifier(epochs=EPOCHS, verbose=False)
    m.fit(X_fit, y_fit)
    return m.predict(X_eval)


def run_xanfis_bio(X_fit, y_fit, X_eval):
    m = BioAnfisClassifier(optim_params={"epoch": EPOCHS}, verbose=False)
    m.fit(X_fit, y_fit)
    return m.predict(X_eval)


import torch
import torch.nn as nn
from sanfis import SANFIS

membfuncs_template = [
    {"function": "gaussian", "n_memb": N_MF_PER_INPUT,
     "params": {"mu": {"value": [-1.0, 1.0], "trainable": True},
                "sigma": {"value": [1.0, 1.0], "trainable": True}}}
    for _ in FEATURES
]


def run_sanfis(X_fit, y_fit, X_eval):
    # epochs=100 je isto sto i default biblioteke, ali ga navodim eksplicitno
    torch.manual_seed(42)
    Xt = torch.tensor(X_fit, dtype=torch.float32)
    yt = torch.tensor(y_fit.reshape(-1, 1), dtype=torch.float32)
    model = SANFIS(membfuncs_template, n_input=len(FEATURES), scale="Std")
    opt = torch.optim.Adam(model.parameters())
    n_val = max(1, int(0.1 * len(Xt)))
    model.fit([Xt[:-n_val], yt[:-n_val]], [Xt[-n_val:], yt[-n_val:]],
              opt, nn.MSELoss(), epochs=EPOCHS, disable_output=True)
    Xe = torch.tensor(X_eval, dtype=torch.float32)
    pred = model.predict(Xe).numpy().ravel()
    return (pred >= 0.5).astype(int)


# treca nezavisna implementacija - klasican Jang (1993) ANFIS, cist Python/
# numpy (twmeggs/anfis), isto kao u prvom eksperimentu - vidi opsiran
# komentar u klasifikacija_eksperiment.py zasto se instalira sa GitHub-a i
# zasto se njena fascikla rucno ubacuje u sys.path (bag u njenom sopstvenom
# __init__.py, plus isto ime "anfis" kao gilardi-jev lokalni fajl)
import importlib.util

# np.product je stara biblioteka ugradila pozivajuci se na numpy alias koji
# je u novijim numpy verzijama ukinut (zamenjen sa np.prod) - vracam ga rucno
if not hasattr(np, "product"):
    np.product = np.prod

_twm_spec = importlib.util.find_spec("anfis")
_twm_pkg_dir = str(Path(_twm_spec.origin).parent)
sys.path.insert(0, _twm_pkg_dir)
import anfis as twm_anfis
import membership.membershipfunction
import membership.mfDerivs


def _twm_backprop_patched(ANFISObj, columnX, columns, theWSum, theW, theLayerFive):
    # original backprop() iz biblioteke, sa jednim ispravljenim redom - vidi
    # komentar u klasifikacija_eksperiment.py za objasnjenje
    paramGrp = [0] * len(ANFISObj.memFuncs[columnX])
    for MF in range(len(ANFISObj.memFuncs[columnX])):
        parameters = np.empty(len(ANFISObj.memFuncs[columnX][MF][1]))
        timesThru = 0
        for alpha in sorted(ANFISObj.memFuncs[columnX][MF][1].keys()):
            bucket3 = np.empty(len(ANFISObj.X))
            for rowX in range(len(ANFISObj.X)):
                varToTest = ANFISObj.X[rowX, columnX]
                tmpRow = np.empty(len(ANFISObj.memFuncs))
                tmpRow.fill(varToTest)
                bucket2 = np.empty(ANFISObj.Y.ndim)
                for colY in range(ANFISObj.Y.ndim):
                    rulesWithAlpha = np.array(np.where(ANFISObj.rules[:, columnX] == MF))[0]
                    adjCols = np.delete(columns, columnX)
                    senSit = membership.mfDerivs.partial_dMF(
                        ANFISObj.X[rowX, columnX], ANFISObj.memFuncs[columnX][MF], alpha)
                    dW_dAplha = senSit * np.array([
                        np.prod([ANFISObj.memClass.evaluateMF(tmpRow)[c][ANFISObj.rules[r][c]] for c in adjCols])
                        for r in rulesWithAlpha])
                    bucket1 = np.empty(len(ANFISObj.rules[:, 0]))
                    for consequent in range(len(ANFISObj.rules[:, 0])):
                        fConsequent = np.dot(
                            np.append(ANFISObj.X[rowX, :], 1.),
                            ANFISObj.consequents[((ANFISObj.X.shape[1] + 1) * consequent):
                                                  (((ANFISObj.X.shape[1] + 1) * consequent) + (ANFISObj.X.shape[1] + 1)), colY])
                        acum = 0
                        if consequent in rulesWithAlpha:
                            acum = dW_dAplha[np.where(rulesWithAlpha == consequent)][0] * theWSum[rowX]
                        acum = acum - theW[consequent, rowX] * np.sum(dW_dAplha)
                        acum = acum / theWSum[rowX] ** 2
                        bucket1[consequent] = fConsequent * acum
                    sum1 = np.sum(bucket1)
                    if ANFISObj.Y.ndim == 1:
                        bucket2[colY] = sum1 * (ANFISObj.Y[rowX] - theLayerFive[rowX, colY]) * (-2)
                    else:
                        bucket2[colY] = sum1 * (ANFISObj.Y[rowX, colY] - theLayerFive[rowX, colY]) * (-2)
                sum2 = np.sum(bucket2)
                bucket3[rowX] = sum2
            sum3 = np.sum(bucket3)
            parameters[timesThru] = sum3
            timesThru = timesThru + 1
        paramGrp[MF] = parameters
    return paramGrp


twm_anfis.backprop = _twm_backprop_patched


def run_twmeggs(X_fit, y_fit, X_eval):
    mins = X_fit.min(axis=0)
    maxs = X_fit.max(axis=0)
    ranges = maxs - mins
    mf = []
    for i in range(X_fit.shape[1]):
        sigma = ranges[i] / N_MF_PER_INPUT if ranges[i] > 0 else 1.0
        mus = np.linspace(mins[i], maxs[i], N_MF_PER_INPUT)
        mf.append([["gaussmf", {"mean": float(mu), "sigma": float(sigma)}] for mu in mus])
    mfc = membership.membershipfunction.MemFuncs(mf)

    model = twm_anfis.ANFIS(X_fit, y_fit, mfc)
    model.trainHybridJangOffLine(epochs=EPOCHS)
    pred = twm_anfis.predict(model, X_eval)
    return (np.asarray(pred).ravel() >= 0.5).astype(int)


sys.path.remove(_twm_pkg_dir)
sys.modules.pop("anfis", None)
sys.modules.pop("membership", None)
sys.modules.pop("membership.membershipfunction", None)
sys.modules.pop("membership.mfDerivs", None)


sys.path.insert(0, str(BASE_DIR / "ANFIS" / "Code_Python"))
import anfis as anf
import utils as utl
import pso as pso_mod


def run_gilardi(X_fit, y_fit, X_eval):
    np.random.seed(42)
    n_mf = [N_MF_PER_INPUT] * len(FEATURES)
    nPop = 40

    Xn_fit, norm_param = utl.normalize_data(X_fit)
    Xn_eval = utl.normalize_data(X_eval, norm_param)
    LB, UB = utl.bounds_pso(Xn_fit, n_mf, 1)
    Yn_fit, scal_param = utl.scale_data(y_fit.reshape(-1, 1).astype(float))
    y_min, y_max = scal_param
    learners = [anf.ANFIS(n_mf=n_mf, n_outputs=1) for _ in range(nPop)]

    def interface_pso(theta, args):
        Xn, Yn, lrns = args
        J = np.zeros(theta.shape[0])
        for i in range(theta.shape[0]):
            J[i] = lrns[i].create_model(theta[i, :], (Xn, Yn))
        return J

    theta, info = pso_mod.PSO(interface_pso, LB, UB, nPop=nPop, epochs=EPOCHS,
                               args=(Xn_fit, Yn_fit, learners))
    best = learners[info[1]]
    pred = best.eval_data(Xn_eval)
    pred = 0.5 * (np.asarray(pred).ravel() + 1.0) * (y_max - y_min) + y_min
    return (pred >= 0.5).astype(int)


import io
import contextlib

_jfp_dir = str(BASE_DIR / "anfis-pytorch")
sys.path.insert(0, _jfp_dir)
sys.modules.pop("anfis", None)
import anfis as jfp_anfis
from membership import make_gauss_mfs
from experimental import train_anfis_with
sys.path.remove(_jfp_dir)
sys.modules.pop("anfis", None)


def _jfp_fit_coeff_patched(self, x, weights, y_actual):
    x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
    weighted_x = torch.einsum('bp, bq -> bpq', weights, x_plus)
    weighted_x[weighted_x == 0] = 1e-12
    weighted_x_2d = weighted_x.view(weighted_x.shape[0], -1)
    y_actual_2d = y_actual.view(y_actual.shape[0], -1)
    sol = torch.linalg.lstsq(weighted_x_2d, y_actual_2d).solution
    self.coeff = sol.view(weights.shape[1], x.shape[1] + 1, -1).transpose(1, 2)


jfp_anfis.ConsequentLayer.fit_coeff = _jfp_fit_coeff_patched


def run_jfpower(X_fit, y_fit, X_eval):
    torch.manual_seed(42)
    Xt = torch.tensor(X_fit, dtype=torch.float32)
    yt = torch.tensor(y_fit.reshape(-1, 1), dtype=torch.float32)

    mins, _ = torch.min(Xt, dim=0)
    maxs, _ = torch.max(Xt, dim=0)
    ranges = maxs - mins
    invars = []
    for i, feat in enumerate(FEATURES):
        sigma = ranges[i] / N_MF_PER_INPUT if ranges[i] > 0 else torch.tensor(1.0)
        mulist = torch.linspace(mins[i], maxs[i], N_MF_PER_INPUT).tolist()
        invars.append((feat, make_gauss_mfs(sigma, mulist)))
    model = jfp_anfis.AnfisNet("klasifikacija", invars, ["BOLEST"], hybrid=True)

    ds = torch.utils.data.TensorDataset(Xt, yt)
    loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)
    criterion = nn.MSELoss(reduction="sum")

    # optimizator i loss nisu moj izbor po slobodnoj proceni - jfpower-ova
    # AnfisNet klasa nema sopstveni ugradjen optimizator, a ovo je tacno ono
    # sto njena sopstvena train_anfis funkcija interno koristi (provereno u
    # izvornom kodu experimental.py, vidi klasifikacija_eksperiment.py)
    with contextlib.redirect_stdout(io.StringIO()):
        train_anfis_with(model, loader, optimizer, criterion, epochs=EPOCHS, show_plots=False)

    Xe = torch.tensor(X_eval, dtype=torch.float32)
    with torch.no_grad():
        pred = model(Xe).numpy().ravel()
    return (pred >= 0.5).astype(int)


import neuro_fuzzy_toolbox as nft


def run_h_anfis(X_fit, y_fit, X_eval):
    # isto kao u prvom eksperimentu
    torch.manual_seed(42)
    Xt = torch.tensor(X_fit, dtype=torch.float32)
    yt = torch.tensor(y_fit, dtype=torch.float32)

    model = nft.h_ANFIS(input_size=len(FEATURES), num_mfs=N_MF_PER_INPUT,
                         outputs=1, output_type='sigmoid', features=FEATURES)
    model.init_premises(Xt)
    model.init_consequents(Xt, yt)

    ds = torch.utils.data.TensorDataset(Xt, yt)
    loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=True)
    trainer = nft.Basic_optimizer_training_algorithm(epochs=EPOCHS, loss_function=nn.BCELoss())
    with contextlib.redirect_stdout(io.StringIO()):
        # val_loader je opciona i ovde se ne koristi early stopping, pa
        # izostavljanje validacionog loadera ne menja treniranje - vidi
        # komentar u klasifikacija_eksperiment.py
        trainer(model, loader)

    Xe = torch.tensor(X_eval, dtype=torch.float32)
    return model.predict(Xe).numpy().ravel().astype(int)


METHODS = {
    "xanfis - classic hybrid (Adam)": run_xanfis_hybrid,
    "xanfis - gradient-only": run_xanfis_gradient,
    "xanfis - bio-inspired": run_xanfis_bio,
    "sanfis": run_sanfis,
    "twmeggs/anfis": run_twmeggs,
    "gabrielegilardi/ANFIS (PSO)": run_gilardi,
    "jfpower/anfis-pytorch": run_jfpower,
    "h_ANFIS (neuro-fuzzy-toolbox)": run_h_anfis,
}


def compute_metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def safe_run(name, fn, X_fit, y_fit, X_eval, y_eval, context):
    # bio-inspired metoda ume povremeno da izbaci NaN/Inf na manjim foldovima
    # - umesto da cela skripta pukne, taj pokusaj se belezi kao neuspeo
    t0 = time.time()
    try:
        y_pred = fn(X_fit, y_fit, X_eval)
        row = compute_metrics(y_eval, y_pred)
    except Exception as e:
        print(f"    [!] {name} nije uspeo ({context}): {e} - belezim kao neuspelo")
        row = {"MSE": np.nan, "accuracy": np.nan, "precision": np.nan,
               "recall": np.nan, "F1": np.nan}
    row["vreme"] = time.time() - t0
    return row


# 10-fold stratifikovana kros validacija na train skupu
print("Pokrecem 10-fold kros validaciju na train skupu (eksperiment 2)...")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2026)
fold_rows = []

for fold_i, (idx_fit, idx_val) in enumerate(skf.split(X_train_raw, y_train), start=1):
    scaler = StandardScaler().fit(X_train_raw[idx_fit])
    Xf = scaler.transform(X_train_raw[idx_fit])
    Xv = scaler.transform(X_train_raw[idx_val])
    yf, yv = y_train[idx_fit], y_train[idx_val]

    for name, fn in METHODS.items():
        row = {"metoda": name, "fold": fold_i}
        row.update(safe_run(name, fn, Xf, yf, Xv, yv, f"fold {fold_i}"))
        fold_rows.append(row)

    print(f"  fold {fold_i}/10 gotov")

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(BASE_DIR / "cv_rezultati_po_foldu_2.csv", index=False)

# agregacija
summary_rows = []
metric_cols = ["MSE", "accuracy", "precision", "recall", "F1"]
for name in METHODS:
    sub = fold_df[fold_df["metoda"] == name]
    row = {"metoda": name}
    for m in metric_cols:
        vals = sub[m].values
        vals = vals[~np.isnan(vals)]  # izbaci neuspele pokusaje iz racuna
        n_ok = len(vals)
        if n_ok == 0:
            mean = var = ci_lo = ci_hi = np.nan
        else:
            mean, var = vals.mean(), (vals.var(ddof=1) if n_ok > 1 else 0.0)
            se = vals.std(ddof=1) / np.sqrt(n_ok) if n_ok > 1 else 0.0
            ci_lo, ci_hi = stats.t.interval(0.95, df=n_ok - 1, loc=mean, scale=se) if se > 0 else (mean, mean)
        row[f"{m}_mean"] = mean
        row[f"{m}_var"] = var
        row[f"{m}_CI95_lo"] = ci_lo
        row[f"{m}_CI95_hi"] = ci_hi
        row[f"{m}_n_uspesnih"] = n_ok
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(BASE_DIR / "cv_rezultati_summary_2.csv", index=False)

# finalni trening na celom (cistom) train skupu, evaluacija na test skupu
print("Finalni trening na celom train skupu, evaluacija na test skupu (eksperiment 2)...")
final_scaler = StandardScaler().fit(X_train_raw)
X_train_s = final_scaler.transform(X_train_raw)
X_test_s = final_scaler.transform(X_test_raw)

test_rows = []
for name, fn in METHODS.items():
    row = {"metoda": name}
    row.update(safe_run(name, fn, X_train_s, y_train, X_test_s, y_test, "finalni test"))
    test_rows.append(row)
    print(f"  {name}: gotovo")

test_df_results = pd.DataFrame(test_rows)
test_df_results.to_csv(BASE_DIR / "test_rezultati_2.csv", index=False)

print("=" * 90)
print("Sacuvano (eksperiment 2):")
for f in ["cv_rezultati_po_foldu_2.csv", "cv_rezultati_summary_2.csv", "test_rezultati_2.csv"]:
    print(" -", f)