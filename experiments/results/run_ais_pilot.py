"""
run_ais_pilot.py
================
Uruchom ten plik ZAMIAST całego marlin_ad_experiments_v2.py
żeby ponownie wykonać tylko sekcję AIS z prawdziwymi danymi DMA.

Użycie:
    python run_ais_pilot.py                          # automatyczne szukanie pliku
    python run_ais_pilot.py aisdk-2025-02-27.csv     # podaj nazwę wprost

Wynik: results_v2/ais_pilot.json  +  results_v2/ais_pilot_figure.png
"""

import sys, os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest

RESULTS_DIR = Path("results_v2")
RESULTS_DIR.mkdir(exist_ok=True)

# ── 1. Znajdź plik DMA ──────────────────────────────────────────────────────
def find_dma_csv(cli_arg=None):
    """Szuka pliku AIS DMA w bieżącym katalogu."""
    candidates = []
    if cli_arg:
        candidates.append(cli_arg)

    # Automatyczne wzorce nazw z portalu dma.dk
    import glob
    candidates += sorted(glob.glob("aisdk*.csv"))
    candidates += sorted(glob.glob("aisdk*.zip"))
    candidates += [
        "aisdk-2025-02-27.csv",
        "aisdk-2024-02-27.csv",
        "aisdk-2023-02-15.csv",
    ]

    print("Szukam pliku AIS DMA...")
    for name in candidates:
        p = Path(name)
        if p.exists():
            size_mb = p.stat().st_size / 1e6
            print(f"  ✓ Znaleziono: {p}  ({size_mb:.1f} MB)")
            return p
        else:
            print(f"  ✗ Brak: {name}")

    return None


# ── 2. Wczytaj dane DMA ─────────────────────────────────────────────────────
def load_dma(csv_path):
    print(f"\nWczytywanie {csv_path} ...")

    # Wczytaj tylko potrzebne kolumny (szybko i bez błędów dtype)
    needed = ["MMSI", "Type of mobile", "SOG", "COG", "Timestamp"]
    df = pd.read_csv(
        csv_path,
        usecols=needed,
        on_bad_lines="skip",
        low_memory=False,
    )
    print(f"  Wczytano {len(df):,} wierszy przed filtrowaniem")

    # Konwersja numeryczna
    df["SOG"] = pd.to_numeric(df["SOG"], errors="coerce")
    df["COG"] = pd.to_numeric(df["COG"], errors="coerce")

    # Tylko statki handlowe
    if "Type of mobile" in df.columns:
        df = df[df["Type of mobile"].isin(["Class A", "Class B"])]
        print(f"  Po filtrze Class A/B: {len(df):,} wierszy")

    # Podstawowe filtry
    df = df.dropna(subset=["SOG", "COG"])
    df = df.query("0 <= SOG <= 30 and 0 <= COG <= 360")
    print(f"  Po filtrze SOG/COG: {len(df):,} wierszy")

    # course_change PER MMSI (kluczowa poprawka)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
        df = df.sort_values(["MMSI", "Timestamp"])

    df["course_change"] = (
        df.groupby("MMSI")["COG"]
        .diff()
        .abs()
        .fillna(0)
        .clip(0, 10)
    )

    X = df[["SOG", "course_change"]].values[:200_000]
    print(f"  Finalne próbki użyte: {len(X):,}")
    return X, "DMA AIS " + csv_path.name


# ── 3. Proxy syntetyczny (fallback) ─────────────────────────────────────────
def make_proxy():
    print("  → Generuję proxy syntetyczny")
    rng = np.random.default_rng(42)
    n = 200_000
    sog = np.clip(rng.normal(8.5, 4.2, n), 0, 30)
    cog_change = np.clip(rng.exponential(0.8, n), 0, 10)
    return np.column_stack([sog, cog_change]), "synthetic AIS proxy"


# ── 4. Analiza ──────────────────────────────────────────────────────────────
def run_pilot(X_ais, source):
    print(f"\nŹródło: {source}")
    print(f"n={len(X_ais):,}  |  SOG: {X_ais[:,0].mean():.2f}±{X_ais[:,0].std():.2f} kn  |  "
          f"COG-change: {X_ais[:,1].mean():.3f}±{X_ais[:,1].std():.3f} deg")

    # Drift syntetyczny: +3 kn bias SOG, 2.5× zmienność kursu
    rng = np.random.default_rng(99)
    X_drifted = X_ais.copy()
    X_drifted[:, 0] += 3.0 + rng.normal(0, 0.5, len(X_ais))
    X_drifted[:, 1] *= 2.5

    # Isolation Forest
    det = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    det.fit(X_ais)
    scores_ref   = -det.score_samples(X_ais)
    scores_drift = -det.score_samples(X_drifted)

    thresh = float(np.percentile(scores_ref, 95))
    anom_ref   = float((scores_ref   >= thresh).mean())
    anom_drift = float((scores_drift >= thresh).mean())
    factor     = round(anom_drift / max(anom_ref, 1e-9), 2)

    ks_sog, p_sog = ks_2samp(X_ais[:, 0], X_drifted[:, 0])
    ks_cog, p_cog = ks_2samp(X_ais[:, 1], X_drifted[:, 1])

    results = {
        "source":               source,
        "n_samples":            int(len(X_ais)),
        "anomaly_rate_ref":     anom_ref,
        "anomaly_rate_drifted": anom_drift,
        "anomaly_rate_factor":  factor,
        "ks_sog":               float(ks_sog),
        "p_sog":                float(p_sog),
        "ks_cog":               float(ks_cog),
        "p_cog":                float(p_cog),
        "mean_score_ref":       float(scores_ref.mean()),
        "mean_score_drifted":   float(scores_drift.mean()),
        "threshold_95pct":      thresh,
    }

    # Wynik
    print("\n" + "="*55)
    print("WYNIKI AIS PILOT:")
    for k, v in results.items():
        print(f"  {k:30s}: {v}")

    # Wartości do manuskryptu
    print("\n" + "="*55)
    print("DO WKLEJENIA W SEKCJĘ 6.8 MANUSKRYPTU:")
    print(f"  anomaly_rate_drifted = {anom_drift:.1%}   → [VALUE]%")
    print(f"  anomaly_rate_factor  = {factor:.1f}×        → [FACTOR]-fold")
    print(f"  ks_sog               = {ks_sog:.3f}         → [VALUE] (KS statistic)")
    print(f"  ks_cog               = {ks_cog:.3f}         → COG-change KS")
    if "DMA AIS" in source:
        print("\n  ✓ Użyte PRAWDZIWE dane DMA — można usunąć flagę proxy z manuskryptu!")
    else:
        print("\n  ⚠ Użyto PROXY — pobierz plik z dma.dk dla finalnej wersji")

    return results, scores_ref, scores_drift, thresh


# ── 5. Figura ───────────────────────────────────────────────────────────────
def make_figure(X_ais, X_drifted, scores_ref, scores_drift, thresh, results):
    source = results["source"]
    ks_sog = results["ks_sog"]
    anom_ref = results["anomaly_rate_ref"]
    anom_drift = results["anomaly_rate_drifted"]
    factor = results["anomaly_rate_factor"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].hist(X_ais[:, 0],       bins=60, alpha=0.6, label="Reference", color="#4472C4")
    axes[0].hist(X_drifted[:, 0],   bins=60, alpha=0.6, label="Drifted (+3 kn bias)", color="#ED7D31")
    axes[0].set_xlabel("SOG (kn)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"SOG distribution shift  (KS = {ks_sog:.3f},  p < 0.001)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(scores_ref,   bins=60, alpha=0.6, label="Reference", color="#4472C4")
    axes[1].hist(scores_drift, bins=60, alpha=0.6, label="Drifted",   color="#ED7D31")
    axes[1].axvline(thresh, color="red", linestyle="--",
                    label=f"Threshold (95th pct = {thresh:.3f})")
    axes[1].set_xlabel("Anomaly score (Isolation Forest)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(
        f"Anomaly rate:  ref = {anom_ref:.1%}  →  drifted = {anom_drift:.1%}"
        f"  ({factor:.1f}× increase)"
    )
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"MARLIN-AD AIS pilot study — {source}", fontsize=11)
    plt.tight_layout()

    fig_path = RESULTS_DIR / "ais_pilot_figure.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigura zapisana: {fig_path}")


# ── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cli_arg = sys.argv[1] if len(sys.argv) > 1 else None

    dma_csv = find_dma_csv(cli_arg)

    if dma_csv:
        try:
            X_ais, source = load_dma(dma_csv)
        except Exception as e:
            print(f"  Błąd wczytywania: {e}")
            print("  → Fallback do proxy syntetycznego")
            X_ais, source = make_proxy()
    else:
        print("Pliku DMA nie znaleziono — używam proxy syntetycznego.")
        X_ais, source = make_proxy()

    results, scores_ref, scores_drift, thresh = run_pilot(X_ais, source)

    X_drifted = X_ais.copy()
    X_drifted[:, 0] += 3.0
    X_drifted[:, 1] *= 2.5

    make_figure(X_ais, X_drifted, scores_ref, scores_drift, thresh, results)

    # Zapisz JSON
    json_path = RESULTS_DIR / "ais_pilot.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON zapisany: {json_path}")
