"""Red neuronal en PyTorch con interfaz de scikit-learn (fit / predict_proba).

- Entrena en GPU si hay CUDA; si no, en CPU.
- Capas densas + BatchNorm + ReLU + Dropout, salida logística.
- Pérdida: entropía cruzada binaria; optimizador AdamW (con weight decay).
- Early stopping: separa el último 10% de las filas como validación interna
  y se queda con la época de mejor AUC.
"""

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from torch import nn

DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


def _red(n_entradas, capas, dropout):
    bloques, previo = [], n_entradas
    for n in capas:
        bloques += [nn.Linear(previo, n), nn.BatchNorm1d(n), nn.ReLU(), nn.Dropout(dropout)]
        previo = n
    bloques.append(nn.Linear(previo, 1))
    return nn.Sequential(*bloques)


class RedTorch(BaseEstimator, ClassifierMixin):
    def __init__(self, capas=(64, 32), dropout=0.1, learning_rate=1e-3, weight_decay=1e-4,
                 batch_size=1024, max_epocas=100, paciencia=8, semilla=0):
        self.capas = capas
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epocas = max_epocas
        self.paciencia = paciencia
        self.semilla = semilla

    def _tensor(self, X):
        return torch.as_tensor(np.asarray(X, dtype=np.float32), device=DISPOSITIVO)

    def fit(self, X, y):
        torch.manual_seed(self.semilla)
        X, y = self._tensor(X), torch.as_tensor(np.asarray(y, dtype=np.float32), device=DISPOSITIVO)
        corte = int(len(X) * 0.9)  # validación interna = último 10% (orden temporal)
        Xe, ye, Xv, yv = X[:corte], y[:corte], X[corte:], y[corte:].cpu().numpy()

        self.modelo_ = _red(X.shape[1], self.capas, self.dropout).to(DISPOSITIVO)
        opt = torch.optim.AdamW(self.modelo_.parameters(), lr=self.learning_rate,
                                weight_decay=self.weight_decay)
        perdida = nn.BCEWithLogitsLoss()
        gen = torch.Generator(device=DISPOSITIVO).manual_seed(self.semilla)

        mejor_auc, mejor_estado, sin_mejora = -1.0, None, 0
        for epoca in range(self.max_epocas):
            self.modelo_.train()
            orden = torch.randperm(len(Xe), device=DISPOSITIVO, generator=gen)
            for i in range(0, len(Xe), self.batch_size):
                idx = orden[i:i + self.batch_size]
                if len(idx) < 2:  # BatchNorm necesita al menos 2 filas
                    continue
                opt.zero_grad()
                perdida(self.modelo_(Xe[idx]).squeeze(1), ye[idx]).backward()
                opt.step()

            auc = roc_auc_score(yv, self._predecir(Xv))
            if auc > mejor_auc:
                mejor_auc, sin_mejora, self.epocas_ = auc, 0, epoca + 1
                mejor_estado = {k: v.detach().clone() for k, v in self.modelo_.state_dict().items()}
            else:
                sin_mejora += 1
                if sin_mejora >= self.paciencia:
                    break
        self.modelo_.load_state_dict(mejor_estado)
        self.classes_ = np.array([0, 1])
        return self

    @torch.no_grad()
    def _predecir(self, X):
        self.modelo_.eval()
        X = X if isinstance(X, torch.Tensor) else self._tensor(X)
        salidas = [torch.sigmoid(self.modelo_(X[i:i + 65536]).squeeze(1)) for i in range(0, len(X), 65536)]
        return torch.cat(salidas).cpu().numpy()

    def predict_proba(self, X):
        p = self._predecir(X)
        return np.column_stack([1 - p, p])
