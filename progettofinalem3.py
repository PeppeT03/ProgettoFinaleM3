# ============================================================
# PROGETTO FINALE M3 - MACHINE LEARNING PER LA REGRESSIONE
# Dataset: Diabetes (Scikit-learn)
#
# Obiettivo:
#   1. Esplorare e controllare il dataset.
#   2. Confrontare sei modelli di regressione con K-Fold CV.
#   3. Ottimizzare Lasso con GridSearchCV.
#   4. Valutare il modello scelto sul Test Set.
#   5. Analizzare le learning curves.
#   6. Visualizzare una regressione nel piano delle prime
#      due componenti principali (PCA).
#
# Nota:
#   I grafici vengono salvati nella cartella "output_m3".
#   La cartella viene creata automaticamente se non esiste.
# ============================================================


# ============================================================
# 1. LIBRERIE
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

from sklearn.datasets import load_diabetes

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV,
    learning_curve
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 2. CARTELLA DI OUTPUT
# ============================================================

# Tutti i grafici prodotti dal programma saranno raccolti qui,
# così la cartella principale del progetto rimane ordinata.

OUTPUT_DIR = Path("output_m3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. CARICAMENTO DEL DATASET E CONTROLLI INIZIALI
# ============================================================

# Carichiamo il dataset Diabetes incluso in Scikit-learn.
# as_frame=True restituisce X e y in formato Pandas.

data = load_diabetes(as_frame=True)

# X contiene le 10 caratteristiche dei pazienti.
# y contiene il valore numerico che vogliamo prevedere.

X = data.data
y = data.target

# Creiamo un DataFrame completo aggiungendo il target.
# Il DataFrame viene usato per l'analisi esplorativa.

df = X.copy()
df["target"] = y

# Prime righe, statistiche descrittive e struttura del dataset.
print("\n========== DATASET DIABETES ==========")
print("Dimensioni delle caratteristiche:", X.shape)
print("Dimensioni del target:", y.shape)
print("\nPrime cinque righe:")
print(df.head())

print("\nStatistiche descrittive:")
print(df.describe())

print("\nInformazioni sulle colonne:")
df.info()

# Controlliamo la presenza di valori mancanti.
print("\nValori mancanti per colonna:")
print(df.isna().sum())


# ============================================================
# 4. ANALISI NUMERICA: CORRELAZIONI E OUTLIER
# ============================================================

# 4A. CORRELAZIONE CON IL TARGET
#
# Calcoliamo la correlazione lineare tra ogni caratteristica
# e il target. Ordiniamo in base al valore assoluto, così
# consideriamo sia le relazioni positive sia quelle negative.
# La correlazione descrive un'associazione, non dimostra causalità.

correlations = df.corr(numeric_only=True)["target"].drop("target")
correlations = correlations.sort_values(key=np.abs, ascending=False)

print("\n========== CORRELAZIONE CON IL TARGET ==========")
print(correlations)

# Selezioniamo le cinque caratteristiche con correlazione
# assoluta più elevata, da usare SOLO nella scatter matrix.
# Tutte le 10 caratteristiche rimangono in X e nei modelli.

top_features = correlations.abs().head(5).index.tolist()
scatter_df = df[top_features + ["target"]]

print("\nCaratteristiche selezionate per la scatter matrix:")
print(top_features)


# 4B. CONTROLLO OUTLIER CON IL METODO IQR

outliers_results = []

for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = (df[col] < lower_limit) | (df[col] > upper_limit)

    outliers_results.append({
        "Variabile": col,
        "Numero outlier": int(outliers.sum())
    })

outliers_df = pd.DataFrame(outliers_results)

print("\n========== OUTLIER INDIVIDUATI CON IQR ==========")
print(outliers_df.to_string(index=False))


# ============================================================
# 5. DIVISIONE TRAINING SET / TEST SET
# ============================================================

# Separiamo i dati in:
# - Training Set (80%): usato per addestramento, CV e tuning.
# - Test Set (20%): tenuto da parte per la valutazione finale.


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== TRAIN / TEST SPLIT ==========")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# ============================================================
# 6. CONFIGURAZIONE K-FOLD CROSS-VALIDATION
# ============================================================

# Dividiamo il Training Set in cinque fold.
# Ogni fold viene usato una volta per la validazione,
# mentre gli altri quattro servono per l'addestramento.
# Il Test Set non viene utilizzato in questa fase.

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nNumero di Fold:", kf.get_n_splits())


# ============================================================
# 7. DEFINIZIONE DEI SEI MODELLI DI REGRESSIONE
# ============================================================

# Per Ridge, Lasso, KNN e SVR inseriamo lo scaler in una Pipeline.
# Durante la Cross-Validation lo scaler viene adattato solo sui
# dati di training di ciascun fold, evitando data leakage.

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(random_state=42),

    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "Lasso": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1, max_iter=10000))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=5))
    ]),

    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf"))
    ])
}

print("\n========== MODELLI DA CONFRONTARE ==========")
for name in models:
    print(name)


# ============================================================
# 8. CONFRONTO DEI MODELLI CON K-FOLD CROSS-VALIDATION
# ============================================================

# cross_val_score valuta ogni modello sui cinque fold.
# Usiamo neg_mean_squared_error: Scikit-learn restituisce il
# MSE con segno negativo, perché per la libreria un punteggio
# più alto deve indicare una prestazione migliore.
# Cambiamo quindi segno per ottenere il normale MSE positivo.

results = []

for name, model in models.items():
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring="neg_mean_squared_error"
    )

    mse_scores = -cv_scores

    results.append({
        "Modello": name,
        "MSE medio": mse_scores.mean(),
        "Deviazione standard": mse_scores.std()
    })

# Creiamo una tabella e ordiniamo i modelli dal MSE più basso.


results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="MSE medio",
    ascending=True
).reset_index(drop=True)

print("\n========== CONFRONTO MODELLI - K-FOLD ==========")
print(results_df.round(2).to_string(index=False))


# ============================================================
# 9. OTTIMIZZAZIONE LASSO CON GRIDSEARCHCV
# ============================================================

# GridSearchCV proverà 50 valori di alpha per Lasso.
# La chiave model__alpha indica il parametro alpha del passaggio
# chiamato "model" nella Pipeline di Lasso.

param_grid = {
    "model__alpha": np.logspace(-4, 1, 50)
}

grid_search = GridSearchCV(
    estimator=models["Lasso"],
    param_grid=param_grid,
    cv=kf,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

# La ricerca viene eseguita esclusivamente sul Training Set.


grid_search.fit(X_train, y_train)

print("\n========== GRID SEARCH - LASSO ==========")
print("Migliori parametri:", grid_search.best_params_)
print("Miglior MSE medio CV:", -grid_search.best_score_)

# Recuperiamo il modello Lasso con il miglior alpha.
best_model = grid_search.best_estimator_


# ============================================================
# 10. VALUTAZIONE FINALE SUL TEST SET
# ============================================================

# Solo adesso utilizziamo il Test Set, rimasto escluso dalle
# fasi di addestramento, Cross-Validation e ottimizzazione.

y_pred_test = best_model.predict(X_test)

mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\n========== VALUTAZIONE FINALE - TEST SET ==========")
print(f"MSE Test: {mse_test:.2f}")
print(f"R2 Test: {r2_test:.4f}")


# ============================================================
# 11. CALCOLO LEARNING CURVE
# ============================================================

# La Learning Curve mostra come cambiano gli errori quando
# aumentiamo progressivamente il numero di campioni usati
# per addestrare il modello.
# Per ogni quantità di dati vengono eseguiti cinque fold.

train_sizes, train_scores, validation_scores = learning_curve(
    estimator=best_model,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=kf,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

# Convertiamo i Negative MSE in valori positivi e calcoliamo
# la media sui cinque fold.

train_mean = -train_scores.mean(axis=1)
validation_mean = -validation_scores.mean(axis=1)

# Calcoliamo anche la deviazione standard

train_std = train_scores.std(axis=1)
validation_std = validation_scores.std(axis=1)

print("\n========== LEARNING CURVE ==========")
print("Forma train_sizes:", train_sizes.shape)
print("Forma train_scores:", train_scores.shape)
print("Forma validation_scores:", validation_scores.shape)


# ============================================================
# 12. PCA E MODELLO DI REGRESSIONE A DUE COMPONENTI
# ============================================================

# La PCA viene adattata SOLO sul Training Set.
# Il Test Set viene trasformato usando le componenti già apprese,
# senza ricalcolare la PCA: in questo modo evitiamo data leakage.

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("\n========== PCA ==========")
print("Dimensioni Training PCA:", X_train_pca.shape)
print("Dimensioni Test PCA:", X_test_pca.shape)
print(f"Varianza spiegata da PC1: {pca.explained_variance_ratio_[0]:.4f}")
print(f"Varianza spiegata da PC2: {pca.explained_variance_ratio_[1]:.4f}")
print(f"Varianza complessiva spiegata: {pca.explained_variance_ratio_.sum():.4f}")

# Questo modello è dedicato alla visualizzazione del piano 3D.
# È distinto dal Lasso ottimizzato usato per la valutazione finale.

model_pca = LinearRegression()
model_pca.fit(X_train_pca, y_train)

# Definiamo i limiti della griglia usando l'intervallo osservato
# nel Training Set e aggiungiamo un piccolo margine del 5%.

pc1_min = X_train_pca[:, 0].min()
pc1_max = X_train_pca[:, 0].max()
pc2_min = X_train_pca[:, 1].min()
pc2_max = X_train_pca[:, 1].max()

margin_pc1 = (pc1_max - pc1_min) * 0.05
margin_pc2 = (pc2_max - pc2_min) * 0.05

pc1_min -= margin_pc1
pc1_max += margin_pc1
pc2_min -= margin_pc2
pc2_max += margin_pc2

# Creiamo 30 coordinate per ciascuna componente e le combiniamo
# in una griglia 30 x 30.

pc1_range = np.linspace(pc1_min, pc1_max, num=30)
pc2_range = np.linspace(pc2_min, pc2_max, num=30)

PC1, PC2 = np.meshgrid(pc1_range, pc2_range)

# Prepariamo ogni posizione della griglia come riga con due
# colonne (PC1 e PC2), calcoliamo il target previsto e riportiamo
# le previsioni alla forma della griglia.

grid_pca = np.c_[PC1.ravel(), PC2.ravel()]
Z_pred = model_pca.predict(grid_pca)
Z_pred_grid = Z_pred.reshape(PC1.shape)


# ============================================================
# 13. GRAFICI E VISUALIZZAZIONI
# ============================================================
# I controlli numerici, il confronto dei modelli e le metriche
# sono stati eseguiti nelle sezioni precedenti. Qui produciamo
# e salviamo separatamente tutte le figure del progetto.


# ------------------------------------------------------------
# 13A. ISTOGRAMMI DI TUTTE LE VARIABILI
# ------------------------------------------------------------

# Ogni colonna del DataFrame viene rappresentata in un istogramma
# separato. Il target ha una scala diversa dalle caratteristiche,
# ma qui viene mostrato in un proprio riquadro.

df.hist(
    bins=20,
    figsize=(14, 10),
    layout=(3, 4)
)

plt.suptitle(
    "Distribuzione delle variabili - Diabetes Dataset",
    fontsize=16
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(
    OUTPUT_DIR / "diabetes_histograms.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ------------------------------------------------------------
# 13B. BOXPLOT DELLE CARATTERISTICHE
# ------------------------------------------------------------

# Escludiamo target dal grafico perché ha una scala numerica
# molto più ampia delle caratteristiche.

plt.figure(figsize=(13, 6))
df.drop(columns="target").boxplot()

plt.title("Boxplot delle caratteristiche - Diabetes")
plt.ylabel("Valori standardizzati")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "diabetes_boxplots_features.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ------------------------------------------------------------
# 13C. BOXPLOT DEL TARGET
# ------------------------------------------------------------

# Il target viene visualizzato separatamente per rendere
# leggibili mediana, quartili, baffi e possibili outlier.

plt.figure(figsize=(7, 5))
df[["target"]].boxplot()

plt.title("Boxplot del target - Diabetes")
plt.ylabel("Progressione della malattia")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "diabetes_boxplot_target.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ------------------------------------------------------------
# 13D. SCATTER MATRIX DELLE CINQUE CARATTERISTICHE PIU'
#      CORRELATE CON IL TARGET + TARGET
# ------------------------------------------------------------

# La matrice completa con 11 variabili produrrebbe 121 grafici,
# difficili da leggere. Per questo mostriamo le cinque variabili
# con maggiore correlazione assoluta con target, più target.
# La selezione serve SOLO alla visualizzazione: X conserva tutte
# e 10 le caratteristiche per l'addestramento dei modelli.

scatter_matrix(
    scatter_df,
    figsize=(12, 12),
    diagonal="hist",
    alpha=0.5,
    hist_kwds={"bins": 15}
)

plt.suptitle(
    "Scatter Matrix - Top 5 Features e Target",
    fontsize=16
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(
    OUTPUT_DIR / "diabetes_scatter_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ------------------------------------------------------------
# 13E. GRAFICO DELLE CORRELAZIONI CON IL TARGET
# ------------------------------------------------------------

# Rappresentiamo le correlazioni ordinate per intensità.
# I valori positivi e negativi mantengono il proprio segno.

plt.figure(figsize=(9, 6))
correlations.sort_values().plot(kind="barh")

plt.title("Correlazione delle caratteristiche con il target")
plt.xlabel("Coefficiente di correlazione")
plt.ylabel("Caratteristica")
plt.axvline(0, color="black", linewidth=0.8)
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "diabetes_target_correlations.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ------------------------------------------------------------
# 13F. GRAFICO DEL NUMERO DI OUTLIER PER VARIABILE
# ------------------------------------------------------------

# Visualizziamo il conteggio ottenuto con la regola IQR.
# Gli outlier vengono segnalati, non eliminati dal dataset.

plt.figure(figsize=(10, 6))
plt.bar(
    outliers_df["Variabile"],
    outliers_df["Numero outlier"]
)

plt.title("Numero di potenziali outlier per variabile - Metodo IQR")
plt.xlabel("Variabile")
plt.ylabel("Numero di outlier")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "diabetes_iqr_outliers.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ------------------------------------------------------------
# 13G. LEARNING CURVE DEL LASSO OTTIMIZZATO
# ------------------------------------------------------------

# La curva blu mostra il MSE medio sui dati di training.
# La curva arancione mostra il MSE medio sui dati di validazione.
# Il confronto aiuta a valutare bias, varianza e generalizzazione.

plt.figure(figsize=(10, 6))

train_line, = plt.plot(
    train_sizes,
    train_mean,
    marker="o",
    label="Training MSE"
)

validation_line, = plt.plot(
    train_sizes,
    validation_mean,
    marker="o",
    label="Validation MSE"
)

# Le fasce trasparenti mostrano la variabilità dei risultati
# tra i cinque fold (media +/- una deviazione standard).
plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    color=train_line.get_color(),
    alpha=0.15
)

plt.fill_between(
    train_sizes,
    validation_mean - validation_std,
    validation_mean + validation_std,
    color=validation_line.get_color(),
    alpha=0.15
)

plt.xlabel("Numero di campioni di addestramento")
plt.ylabel("Mean Squared Error")
plt.title("Learning Curve - Lasso ottimizzato")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "diabetes_learning_curve.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ------------------------------------------------------------
# 13H. PIANO DI REGRESSIONE 3D NELLO SPAZIO PCA
# ------------------------------------------------------------

# I punti rappresentano i pazienti reali del Training Set.
# La superficie rappresenta le previsioni del modello lineare
# addestrato sulle sole componenti PC1 e PC2.
# Questo modello è usato per la visualizzazione, non sostituisce
# il Lasso ottimizzato valutato sul Test Set.

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Superficie del piano di regressione.
surface = ax.plot_surface(
    PC1,
    PC2,
    Z_pred_grid,
    cmap="viridis",
    alpha=0.6,
    edgecolor="none"
)

# Punti osservati: coordinate PCA e target reali.
ax.scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    y_train,
    c=y_train,
    cmap="plasma",
    s=25,
    alpha=0.8
)

ax.set_title("Piano di regressione - PCA Diabetes")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("Target")

fig.colorbar(
    surface,
    ax=ax,
    shrink=0.6,
    label="Target previsto"
)

ax.view_init(elev=25, azim=135)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "diabetes_pca_regression_plane.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 14. RIEPILOGO FINALE
# ============================================================

# Stampiamo i risultati principali in un unico riepilogo,
# utile durante il controllo del progetto e per il README.

print("\n========== RIEPILOGO FINALE DEL PROGETTO ==========")
print("Miglior modello selezionato: Lasso")
print("Miglior alpha:", grid_search.best_params_["model__alpha"])
print(f"MSE medio Cross-Validation: {-grid_search.best_score_:.2f}")
print(f"MSE Test: {mse_test:.2f}")
print(f"R2 Test: {r2_test:.4f}")
print(f"Varianza conservata dalle prime due componenti PCA: {pca.explained_variance_ratio_.sum():.4f}")
print("\nGrafici salvati nella cartella:", OUTPUT_DIR.resolve())

# Fine del Progetto Finale M3.
