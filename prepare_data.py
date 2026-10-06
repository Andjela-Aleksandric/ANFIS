"""
Priprema podataka za ANFIS eksperiment.

Izvor: stock_dataset.csv iz gabrielegilardi/ANFIS repozitorijuma (vec kloniran
u sklopu prethodnog dela rada) - ovo JE originalni UCI "Istanbul Stock Exchange"
dataset (Akbilgic, Bozdogan & Balaban), isti dataset na koji se oslanja rad
(Boyacioglu & Avci, 2010) naveden u pregledu literature. Podaci su vec
pretprocesirani kao dnevni prinosi (returns), 536 trgovackih dana, 3 ulazne
promenljive (prinosi odabranih medjunarodnih berzanskih indeksa) i 2 izlazne
promenljive (prinosi ISE indeksa, TL i USD osnova).

Kolone u izvornom fajlu nisu imenovane, koriste se generički nazivi
Input_1..3 / Output_1..2

Dizajn eksperimenta:
  - cilj (target):            Output_1  (ISE prinos)
  - obicne ulazne promenljive: Input_1, Input_2, Input_3, Output_2
  - "state"/premisa za S-ANFIS eksperiment:
        Regime = predznak Input_1 na ISTOM danu (1 = pozitivan dan na vodecem
        medjunarodnom trzistu, 0 = negativan) - operacionalizacija
        "regime-switching" ideje iz S-ANFIS rada (Lenhard & Maringer, 2022)
        obradjenog u pregledu literature: model uci da li se odnos izmedju
        ulaza i ISE prinosa menja u zavisnosti od toga da li je vodece
        trziste tog dana u rastu ili padu.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "ANFIS" / "Code_Python" / "stock_dataset.csv"

raw = np.loadtxt(DATASET_PATH, delimiter=",")

df = pd.DataFrame(raw, columns=["Input_1", "Input_2", "Input_3", "Output_1", "Output_2"])
df["Regime"] = (df["Input_1"] > 0).astype(int)
df["Day"] = np.arange(len(df))

# ovo je pripremljen dataset koji koristi glavni eksperiment
df.to_csv(BASE_DIR / "ise_features.csv", index=False)

# kontrola veličine skupa i osnovne raspodele podataka
n = len(df)
n_train = int(n * 0.8)
print(f"Ukupno uzoraka: {n}  |  Trening: {n_train}  |  Test: {n - n_train}")
print(df.describe())
print("\nBalans Regime promenljive:")
print(df["Regime"].value_counts())
