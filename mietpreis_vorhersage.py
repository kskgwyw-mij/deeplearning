"""
================================================================================
  DEEP LEARNING - EINFÜHRUNG: Neuronales Netz mit PyTorch
  Use Case: Vorhersage von Mietpreisen (wie auf willhaben.at)
================================================================================

Lernziele:
  1. Was sind Tensors und warum brauchen wir sie?
  2. Wie baut man ein neuronales Netz mit PyTorch?
  3. Was sind Aktivierungsfunktionen?
  4. Was ist ein Loss und wie trainiert man damit?
  5. Was ist Backpropagation?
  6. Wie bewertet man ein trainiertes Modell?

Real-World Kontext:
  Stell dir vor, du willst auf Basis von Wohnungsmerkmalen
  (Größe, Zimmer, Lage) den monatlichen Mietpreis schätzen.
  Genau das lernt unser Netz!
================================================================================
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import torch                          # Das Herzstück: PyTorch selbst
import torch.nn as nn                 # nn = "neural network" Bausteine (Schichten, Loss)
import torch.optim as optim           # Optimierungsalgorithmen (z.B. Adam, SGD)
from torch.utils.data import Dataset, DataLoader  # Datenverwaltung & Batching

import numpy as np                    # Numerische Berechnungen
import matplotlib.pyplot as plt       # Visualisierung
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split  # Daten aufteilen
from sklearn.preprocessing import StandardScaler      # Feature-Normalisierung

# Für reproduzierbare Ergebnisse (gleicher "Zufalls"-Startwert)
torch.manual_seed(42)
np.random.seed(42)


# ── 1. SYNTHETISCHE DATEN ERSTELLEN ──────────────────────────────────────────
"""
In der Praxis würden wir echte Daten von willhaben.at scrapen.
Hier generieren wir realistische synthetische Daten,
die echte Muster nachahmen.

Features (Eingaben):
  - wohnflaeche_m2  : Größe der Wohnung in m²     (z.B. 30–120 m²)
  - zimmeranzahl    : Anzahl der Zimmer             (1–5)
  - stockwerk       : In welchem Stockwerk          (0–10)
  - entfernung_zentrum_km: Entfernung zum Zentrum  (0.5–20 km)
  - baujahr_norm    : Baujahr normiert auf [0,1]    (alt=0, neu=1)

Ziel (Ausgabe):
  - monatliche Miete in EUR
"""

def erzeuge_wohnungsdaten(n_samples: int = 800):
    """
    Erzeugt synthetische Wohnungsdaten mit realistischen Abhängigkeiten.
    
    Args:
        n_samples: Anzahl der Datenpunkte (Wohnungen)
    
    Returns:
        X: Feature-Matrix  (n_samples × 5)
        y: Zielvektor      (n_samples,)  — Mietpreise in EUR
    """
    
    # --- Features zufällig ziehen ---
    wohnflaeche      = np.random.uniform(25, 130, n_samples)     # m²
    zimmeranzahl     = np.random.randint(1, 6, n_samples)        # 1–5 Zimmer
    stockwerk        = np.random.randint(0, 11, n_samples)       # EG–10.OG
    entfernung_km    = np.random.uniform(0.5, 20, n_samples)     # km vom Zentrum
    baujahr_norm     = np.random.uniform(0, 1, n_samples)        # 0=alt, 1=neu

    # --- Mietpreis berechnen (angelehnt an reale Zusammenhänge) ---
    # Jeder Faktor hat einen interpretierbaren Einfluss:
    #   + Fläche          → größer = teurer
    #   + Zimmer          → mehr Zimmer = leicht teurer
    #   + Stockwerk       → höher = etwas teurer (Aussicht)
    #   - Entfernung      → weiter weg = günstiger
    #   + Baujahr         → neuer = teurer
    #   + Rauschen        → echte Daten sind nie perfekt!
    
    miete = (
          8.5  * wohnflaeche         # ~8.50 €/m² Basispreis
        + 50   * zimmeranzahl        # +50 € je Zimmer
        + 15   * stockwerk           # +15 € je Stockwerk
        - 20   * entfernung_km       # -20 € je km Entfernung
        + 100  * baujahr_norm        # bis +100 € für Neubau
        + 200                        # Fixkosten-Offset
        + np.random.normal(0, 80, n_samples)  # Zufallsrauschen
    )
    
    # Miete auf realistische Bandbreite clippen (min 300 €, max 3000 €)
    miete = np.clip(miete, 300, 3000)
    
    # Feature-Matrix zusammenbauen
    X = np.column_stack([
        wohnflaeche, zimmeranzahl, stockwerk, entfernung_km, baujahr_norm
    ])
    
    return X.astype(np.float32), miete.astype(np.float32)


# ── 2. PYTORCH DATASET KLASSE ─────────────────────────────────────────────────
"""
PyTorch arbeitet mit Dataset- und DataLoader-Objekten.
  - Dataset: Kapselt unsere Daten, macht sie indexierbar
  - DataLoader: Teilt Daten in Batches auf, kann sie mischen

Warum Batches?
  Statt alle 800 Wohnungen auf einmal zu verarbeiten (zu viel RAM),
  trainieren wir mit kleinen Paketen (z.B. 32 Wohnungen = 1 Batch).
  Das macht das Training stabiler und schneller.
"""

class WohnungsDatensatz(Dataset):
    """
    Eigenes Dataset für unsere Wohnungsdaten.
    Erbt von torch.utils.data.Dataset und muss 3 Methoden implementieren:
      __init__  → Daten speichern
      __len__   → Anzahl der Datenpunkte zurückgeben
      __getitem__ → Einzelnen Datenpunkt per Index zurückgeben
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Konvertiert NumPy-Arrays zu PyTorch-Tensors.
        
        Tensor = die grundlegende Datenstruktur in PyTorch
                 Ähnlich wie NumPy-Array, aber GPU-fähig und
                 kann Gradienten für Backpropagation speichern.
        """
        self.X = torch.tensor(X, dtype=torch.float32)  # Features als Float-Tensor
        self.y = torch.tensor(y, dtype=torch.float32)  # Zielwerte als Float-Tensor
    
    def __len__(self) -> int:
        """Gibt an, wie viele Wohnungen im Datensatz sind."""
        return len(self.X)
    
    def __getitem__(self, idx: int):
        """Gibt Features und Zielwert für Index idx zurück."""
        return self.X[idx], self.y[idx]


# ── 3. DAS NEURONALE NETZ ─────────────────────────────────────────────────────
"""
Architektur unseres Netzes:

  Eingabe (5 Features)
       │
  ┌────▼────┐
  │ Linear  │  5 → 64 Neuronen   (erste versteckte Schicht)
  │  ReLU   │  Aktivierungsfunktion: max(0, x)
  └────┬────┘
       │
  ┌────▼────┐
  │ Linear  │  64 → 32 Neuronen  (zweite versteckte Schicht)
  │  ReLU   │  
  └────┬────┘
       │
  ┌────▼────┐
  │ Linear  │  32 → 1 Neuron     (Ausgabeschicht: 1 Mietpreis)
  └────┬────┘
       │
  Ausgabe (1 Zahl: vorhergesagter Mietpreis in EUR)

Warum mehrere Schichten?
  Jede Schicht lernt komplexere Muster. Die erste Schicht kombiniert
  Rohfeatures (z.B. "Fläche × Zimmer"), tiefere Schichten abstrahieren
  das weiter (z.B. "große Wohnung in guter Lage").

Warum ReLU?
  Ohne Aktivierungsfunktion wäre das gesamte Netz nur eine einzige
  lineare Gleichung — egal wie viele Schichten. ReLU (Rectified Linear Unit)
  fügt Nichtlinearität ein: f(x) = max(0, x)
  Das erlaubt dem Netz, komplexe, nicht-lineare Zusammenhänge zu lernen.
"""

class MietpreisNetz(nn.Module):
    """
    Neuronales Netz zur Vorhersage von Mietpreisen.
    
    Erbt von nn.Module — der Basisklasse für alle PyTorch-Netze.
    Muss 2 Methoden implementieren:
      __init__  → Schichten definieren
      forward   → Wie fließen Daten durch das Netz (Vorwärtsdurchlauf)
    """
    
    def __init__(self, eingabe_dim: int = 5):
        """
        Definiert alle Schichten des Netzes.
        
        Args:
            eingabe_dim: Anzahl der Eingabe-Features (bei uns 5)
        """
        super().__init__()  # Initialisiert die Elternklasse nn.Module
        
        # nn.Sequential: Schichten werden der Reihe nach ausgeführt
        # Das ist praktisch, damit wir im forward()-Aufruf weniger schreiben müssen
        self.netz = nn.Sequential(
            
            # ── Erste versteckte Schicht ──────────────────────────────────
            # nn.Linear(in, out): Vollverbundene Schicht (jedes Neuron mit jedem)
            # Berechnet: output = X * W^T + b
            #   W = Gewichtsmatrix (wird trainiert)
            #   b = Bias-Vektor    (wird trainiert)
            # in=5 (unsere 5 Features), out=64 (64 Neuronen in dieser Schicht)
            nn.Linear(eingabe_dim, 64),
            
            # Batch Normalization: Normalisiert Aktivierungen innerhalb eines Batches
            # Vorteil: Stabileres Training, schnellere Konvergenz
            nn.BatchNorm1d(64),
            
            # ReLU Aktivierungsfunktion: f(x) = max(0, x)
            # Alle negativen Werte werden zu 0 — das erzeugt Sparsität & Nichtlinearität
            nn.ReLU(),
            
            # Dropout: Schaltet zufällig 20% der Neuronen ab (p=0.2)
            # Das ist Regularisierung: verhindert, dass das Netz "auswendig lernt"
            # (= Overfitting). Nur während dem Training aktiv!
            nn.Dropout(p=0.2),
            
            # ── Zweite versteckte Schicht ─────────────────────────────────
            # Weniger Neuronen (64→32): Das Netz "verdichtet" die Information
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            # ── Ausgabeschicht ────────────────────────────────────────────
            # 32 Eingaben → 1 Ausgabe (der vorhergesagte Mietpreis)
            # KEINE Aktivierungsfunktion hier! Bei Regression wollen wir
            # beliebige reelle Zahlen ausgeben, keine Einschränkung.
            nn.Linear(32, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vorwärtsdurchlauf: Wie fließen Daten durch das Netz?
        
        Args:
            x: Eingabe-Tensor (Batch × 5 Features)
        
        Returns:
            Ausgabe-Tensor (Batch × 1) — vorhergesagte Mietpreise
        """
        ausgabe = self.netz(x)      # Daten durch alle Schichten schicken
        return ausgabe.squeeze(1)   # Form (Batch, 1) → (Batch,) für einfachere Berechnung


# ── 4. TRAINING ───────────────────────────────────────────────────────────────
"""
Der Trainingsloop — das Herzstück des Deep Learning:

Für jeden Epoch (vollständiger Durchlauf durch alle Daten):
  Für jeden Batch (kleines Datenpaket):
    1. VORWÄRTSDURCHLAUF  → Netz macht Vorhersage
    2. LOSS BERECHNEN     → Wie falsch ist die Vorhersage?
    3. RÜCKWÄRTSDURCHLAUF → Gradienten berechnen (Backpropagation)
    4. PARAMETER UPDATE   → Gewichte in Richtung besserer Vorhersage anpassen

Was ist der Loss?
  Eine Zahl, die misst, wie schlecht unser Netz gerade ist.
  MSE (Mean Squared Error) = Durchschnitt der quadrierten Abweichungen
  MSE = (1/n) × Σ (vorhersage - wahrheit)²
  Unser Ziel: Loss minimieren!

Was ist Backpropagation?
  Der Algorithmus, der berechnet, wie viel jedes Gewicht zum Fehler beiträgt
  (= Gradient). Dann werden alle Gewichte leicht in die "bessere" Richtung
  verschoben — das ist Gradient Descent.
"""

def trainiere_modell(modell, train_loader, val_loader, 
                     n_epochs: int = 80, lernrate: float = 0.001):
    """
    Haupttrainingsfunktion.
    
    Args:
        modell      : Unser MietpreisNetz
        train_loader: DataLoader für Trainingsdaten
        val_loader  : DataLoader für Validierungsdaten  
        n_epochs    : Wie oft soll das Netz alle Trainingsdaten sehen?
        lernrate    : Wie groß ist jeder "Lernschritt"? (Hyperparameter!)
    
    Returns:
        train_losses, val_losses: Verlauf des Losses (für Plot)
    """
    
    # ── Loss-Funktion ──────────────────────────────────────────────────────
    # MSELoss = Mean Squared Error für Regressionsprobleme
    # Alternative: L1Loss (weniger empfindlich für Ausreißer), HuberLoss
    kriterium = nn.MSELoss()
    
    # ── Optimierer ────────────────────────────────────────────────────────
    # Adam (Adaptive Moment Estimation) = verbreiteter Optimierer
    # Passt die Lernrate für jeden Parameter automatisch an
    # Alternative: SGD (einfacher, aber braucht mehr Tuning)
    optimierer = optim.Adam(modell.parameters(), lr=lernrate)
    
    # ── Lernraten-Scheduler ───────────────────────────────────────────────
    # Reduziert die Lernrate, wenn der Loss nicht mehr sinkt
    # "Geduld" von 10 Epochs, dann Lernrate × 0.5
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimierer, patience=10, factor=0.5, verbose=False
    )
    
    train_losses = []
    val_losses   = []
    
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>10} | {'RMSE (€)':>10}")
    print("-" * 48)
    
    for epoch in range(n_epochs):
        
        # ══════════════════════════════════════════════
        # TRAININGSPHASE
        # ══════════════════════════════════════════════
        modell.train()   # Wichtig! Aktiviert Dropout & BatchNorm im Trainingsmodus
        
        batch_losses = []
        
        for X_batch, y_batch in train_loader:
            
            # Schritt 1: Gradienten aus dem letzten Schritt auf Null setzen
            # (PyTorch akkumuliert Gradienten standardmäßig — das wollen wir nicht)
            optimierer.zero_grad()
            
            # Schritt 2: Vorwärtsdurchlauf — Vorhersage berechnen
            vorhersagen = modell(X_batch)
            
            # Schritt 3: Loss berechnen — wie schlecht sind wir?
            loss = kriterium(vorhersagen, y_batch)
            
            # Schritt 4: Backpropagation — Gradienten berechnen
            # PyTorch berechnet automatisch dLoss/dW für jedes Gewicht W
            loss.backward()
            
            # Gradient Clipping: Verhindert explodierende Gradienten
            # Keine Gewichtsänderung > 1.0 auf einmal
            nn.utils.clip_grad_norm_(modell.parameters(), max_norm=1.0)
            
            # Schritt 5: Gewichte aktualisieren (Gradient Descent Schritt)
            optimierer.step()
            
            batch_losses.append(loss.item())  # .item() holt den Python-Float aus dem Tensor
        
        # Durchschnittlicher Loss über alle Batches
        train_loss_avg = np.mean(batch_losses)
        
        # ══════════════════════════════════════════════
        # VALIDIERUNGSPHASE
        # ══════════════════════════════════════════════
        modell.eval()   # Deaktiviert Dropout, setzt BatchNorm in Eval-Modus
        
        with torch.no_grad():  # Keine Gradientenberechnung nötig (spart Speicher & Zeit)
            val_losses_batch = []
            for X_val, y_val in val_loader:
                val_vorhersagen = modell(X_val)
                val_loss = kriterium(val_vorhersagen, y_val)
                val_losses_batch.append(val_loss.item())
        
        val_loss_avg = np.mean(val_losses_batch)
        
        # Lernrate anpassen wenn nötig
        scheduler.step(val_loss_avg)
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        
        # Alle 10 Epochs einen Statusbericht ausgeben
        if (epoch + 1) % 10 == 0:
            rmse = np.sqrt(val_loss_avg)  # RMSE in der Original-Einheit (€)
            print(f"{epoch+1:>6} | {train_loss_avg:>12.1f} | {val_loss_avg:>10.1f} | {rmse:>9.1f}€")
    
    return train_losses, val_losses


# ── 5. EVALUATION ─────────────────────────────────────────────────────────────

def evaluiere_modell(modell, scaler_y, X_test_tensor, y_test_tensor):
    """
    Bewertet das trainierte Modell auf den Testdaten.
    
    Wichtige Metriken:
    - RMSE (Root Mean Squared Error): Durchschnittlicher Fehler in € 
      → "Im Schnitt liegen wir X Euro daneben"
    - MAE  (Mean Absolute Error): Robuster gegen Ausreißer
    - R²   (Bestimmtheitsmaß): 1.0 = perfekt, 0.0 = so gut wie Mittelwert
    """
    modell.eval()
    
    with torch.no_grad():
        vorhersagen_scaled = modell(X_test_tensor).numpy()
        wahrheit_scaled    = y_test_tensor.numpy()
    
    # Rücktransformation (wir haben die Zielvariable skaliert)
    vorhersagen = scaler_y.inverse_transform(vorhersagen_scaled.reshape(-1, 1)).flatten()
    wahrheit    = scaler_y.inverse_transform(wahrheit_scaled.reshape(-1, 1)).flatten()
    
    # Metriken berechnen
    rmse = np.sqrt(np.mean((vorhersagen - wahrheit) ** 2))
    mae  = np.mean(np.abs(vorhersagen - wahrheit))
    ss_res = np.sum((wahrheit - vorhersagen) ** 2)
    ss_tot = np.sum((wahrheit - np.mean(wahrheit)) ** 2)
    r2   = 1 - ss_res / ss_tot
    
    print("\n" + "="*48)
    print("  MODELLEVALUATION AUF TESTDATEN")
    print("="*48)
    print(f"  RMSE  : {rmse:>8.1f} €   (mittlerer Fehler)")
    print(f"  MAE   : {mae:>8.1f} €   (mittl. abs. Fehler)")
    print(f"  R²    : {r2:>8.4f}     (Bestimmtheitsmaß)")
    print("="*48)
    
    return vorhersagen, wahrheit, rmse, mae, r2


# ── 6. VISUALISIERUNG ─────────────────────────────────────────────────────────

def erstelle_plots(train_losses, val_losses, vorhersagen, wahrheit):
    """Erstellt aussagekräftige Plots für die Ergebnisse."""
    
    fig = plt.figure(figsize=(16, 10), facecolor='#0f0f1a')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    
    # Farben
    farbe_train  = '#4fc3f7'
    farbe_val    = '#ef9a9a'
    farbe_punkte = '#a5d6a7'
    farbe_linie  = '#ffcc02'
    bg_ax        = '#1a1a2e'
    text_farbe   = '#e8e8f0'
    
    def style_ax(ax, titel):
        ax.set_facecolor(bg_ax)
        ax.set_title(titel, color=text_farbe, fontsize=11, pad=10, fontweight='bold')
        ax.tick_params(colors=text_farbe, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')
        ax.xaxis.label.set_color(text_farbe)
        ax.yaxis.label.set_color(text_farbe)
        ax.grid(True, color='#333355', linewidth=0.5, alpha=0.7)
    
    # ── Plot 1: Loss-Kurven ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, color=farbe_train, linewidth=2, label='Training Loss')
    ax1.plot(epochs, val_losses,   color=farbe_val,   linewidth=2, label='Validierungs Loss', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.legend(facecolor='#1a1a2e', labelcolor=text_farbe, framealpha=0.8)
    style_ax(ax1, '📉 Lernkurve: Loss über Trainingszeit\n(je niedriger, desto besser)')
    
    # ── Plot 2: Netzarchitektur schematisch ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(bg_ax)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('🧠 Netzarchitektur', color=text_farbe, fontsize=11, fontweight='bold', pad=10)
    
    schichten = [
        (1, 8.5, '5\nFeatures', '#4fc3f7', 0.7),
        (3, 7.5, '64\nNeuronen', '#7986cb', 1.0),
        (5, 7.5, '32\nNeuronen', '#7986cb', 0.8),
        (7, 8.5, '1\nAusgabe', '#a5d6a7', 0.5),
    ]
    labels = ['Input\nLayer', 'Hidden\nLayer 1\n+ ReLU', 'Hidden\nLayer 2\n+ ReLU', 'Output\nLayer']
    
    for i, ((x, y, txt, farbe, size), label) in enumerate(zip(schichten, labels)):
        circle = plt.Circle((x, y), size, color=farbe, alpha=0.8)
        ax2.add_patch(circle)
        ax2.text(x, y, txt, ha='center', va='center', fontsize=7, 
                 color='white', fontweight='bold')
        ax2.text(x, y - size - 0.5, label, ha='center', va='top', fontsize=6.5,
                 color=text_farbe, alpha=0.8)
        if i < len(schichten) - 1:
            x_next = schichten[i+1][0]
            ax2.annotate('', xy=(x_next - schichten[i+1][4] - 0.1, schichten[i+1][1]),
                        xytext=(x + size + 0.1, y),
                        arrowprops=dict(arrowstyle='->', color='#ffffff55', lw=1.2))
    
    # ── Plot 3: Vorhersage vs. Wahrheit ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    max_wert = max(vorhersagen.max(), wahrheit.max())
    min_wert = min(vorhersagen.min(), wahrheit.min())
    
    ax3.scatter(wahrheit, vorhersagen, alpha=0.4, s=18, color=farbe_punkte,
                edgecolors='none', label='Wohnungen')
    ax3.plot([min_wert, max_wert], [min_wert, max_wert], 
             color=farbe_linie, linewidth=2, linestyle='--', label='Perfekte Vorhersage')
    ax3.set_xlabel('Tatsächliche Miete (€)')
    ax3.set_ylabel('Vorhergesagte Miete (€)')
    ax3.legend(facecolor='#1a1a2e', labelcolor=text_farbe)
    style_ax(ax3, '🏠 Vorhersage vs. Wahrheit\n(Punkte nahe der Linie = gute Vorhersage)')
    
    # ── Plot 4: Fehler-Histogramm ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    fehler = vorhersagen - wahrheit
    ax4.hist(fehler, bins=30, color=farbe_train, alpha=0.8, edgecolor='#0f0f1a')
    ax4.axvline(0, color=farbe_linie, linewidth=2, linestyle='--', label='Kein Fehler')
    ax4.axvline(fehler.mean(), color=farbe_val, linewidth=1.5, 
                linestyle=':', label=f'Mittelwert: {fehler.mean():.0f}€')
    ax4.set_xlabel('Vorhersagefehler (€)')
    ax4.set_ylabel('Anzahl')
    ax4.legend(facecolor='#1a1a2e', labelcolor=text_farbe, fontsize=8)
    style_ax(ax4, '📊 Fehlerverteilung\n(idealerweise symmetrisch um 0)')
    
    # Haupttitel
    fig.suptitle(
        '🔍 Deep Learning — Mietpreisvorhersage mit PyTorch',
        color=text_farbe, fontsize=14, fontweight='bold', y=0.98
    )
    
    plt.savefig('neural_network_ergebnisse.png', 
                dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    print("\n  📊 Plot gespeichert: neural_network_ergebnisse.png")
    plt.close()


# ── HAUPTPROGRAMM ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  DEEP LEARNING EINFÜHRUNG — MIETPREISVORHERSAGE")
    print("  Verwendete Bibliothek: PyTorch", torch.__version__)
    print("="*60)
    
    # ── Daten generieren ──────────────────────────────────────────────────
    print("\n[1/5] Daten werden generiert...")
    X, y = erzeuge_wohnungsdaten(n_samples=1000)
    print(f"      {len(X)} Wohnungen erzeugt")
    print(f"      Features pro Wohnung: {X.shape[1]}")
    print(f"      Mietpreise: {y.min():.0f}€ – {y.max():.0f}€ (Ø {y.mean():.0f}€)")
    
    # ── Daten aufteilen (80% Training, 10% Validierung, 10% Test) ────────
    print("\n[2/5] Daten aufteilen...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"      Training:    {len(X_train)} Wohnungen")
    print(f"      Validierung: {len(X_val)} Wohnungen")
    print(f"      Test:        {len(X_test)} Wohnungen")
    
    # ── Normalisierung ────────────────────────────────────────────────────
    # Neuronale Netze lernen viel besser, wenn alle Features ähnliche Skalen haben!
    # StandardScaler: transformiert zu Mittelwert=0, Standardabweichung=1
    # WICHTIG: fit() nur auf Trainingsdaten! (kein "data leakage")
    print("\n[3/5] Features normalisieren (StandardScaler)...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_s = scaler_X.fit_transform(X_train)   # fit_transform: lernen + anwenden
    X_val_s   = scaler_X.transform(X_val)          # transform:     nur anwenden
    X_test_s  = scaler_X.transform(X_test)
    
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_s   = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_s  = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # DataLoader erstellen (Batch-Größe = 32)
    train_ds = WohnungsDatensatz(X_train_s.astype(np.float32), y_train_s.astype(np.float32))
    val_ds   = WohnungsDatensatz(X_val_s.astype(np.float32),   y_val_s.astype(np.float32))
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)   # shuffle=Mischen!
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)
    
    # ── Modell, Loss, Optimierer ──────────────────────────────────────────
    print("\n[4/5] Modell wird trainiert...")
    modell = MietpreisNetz(eingabe_dim=5)
    
    # Anzahl der trainierbaren Parameter ausgeben
    n_params = sum(p.numel() for p in modell.parameters() if p.requires_grad)
    print(f"      Trainierbare Parameter: {n_params:,}")
    
    # Training starten
    train_losses, val_losses = trainiere_modell(
        modell, train_loader, val_loader, n_epochs=80, lernrate=0.001
    )
    
    # ── Evaluation ────────────────────────────────────────────────────────
    print("\n[5/5] Modell wird evaluiert...")
    X_test_tensor = torch.tensor(X_test_s.astype(np.float32))
    y_test_tensor = torch.tensor(y_test_s.astype(np.float32))
    
    vorhersagen, wahrheit, rmse, mae, r2 = evaluiere_modell(
        modell, scaler_y, X_test_tensor, y_test_tensor
    )
    
    # ── Beispielvorhersage ────────────────────────────────────────────────
    print("\n  BEISPIELVORHERSAGEN:")
    print(f"  {'Fläche':>7} | {'Zimmer':>6} | {'Stock':>5} | {'Entf.':>6} | {'Bauj.':>6} | {'Wahrheit':>9} | {'Vorhersage':>11}")
    print("  " + "-"*65)
    
    for i in range(5):
        w = wahrheit[i]
        v = vorhersagen[i]
        feats = X_test[i]
        print(f"  {feats[0]:>6.0f}m² | {feats[1]:>5.0f}Zi | {feats[2]:>4.0f}OG | {feats[3]:>5.1f}km | {feats[4]:>5.2f}  | {w:>8.0f}€ | {v:>10.0f}€")
    
    # ── Plots erstellen ───────────────────────────────────────────────────
    erstelle_plots(train_losses, val_losses, vorhersagen, wahrheit)
    
    print("\n✅ Fertig! Das neuronale Netz hat erfolgreich Mietpreise gelernt.")
    print(f"   → Im Schnitt liegt es {rmse:.0f}€ daneben (RMSE)")
    print(f"   → Es erklärt {r2*100:.1f}% der Preisvariation (R²)\n")


if __name__ == "__main__":
    main()
