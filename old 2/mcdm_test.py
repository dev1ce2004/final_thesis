import numpy as np

# -----------------------------
# PODACI
# -----------------------------

alternative_names = ["Ruta A", "Ruta B", "Ruta C"]

# stupci:
# sigurnost       -> benefit (veće je bolje)
# vrijeme         -> cost   (manje je bolje)
# prohodnost      -> benefit (veće je bolje)

X = np.array([
    [0.90, 12, 0.70],
    [0.75,  8, 0.80],
    [0.85, 10, 0.95]
])

# težine kriterija
weights = np.array([
    0.648,   # sigurnost
    0.122,   # vrijeme evakuacije
    0.230    # prohodnost
])

criteria_types = ["benefit", "cost", "benefit"]


# -----------------------------
# TOPSIS
# -----------------------------

def topsis(X, weights, criteria_types):

    # normalizacija
    R = X / np.sqrt((X ** 2).sum(axis=0))

    # ponderiranje
    V = R * weights

    ideal_best = np.zeros(X.shape[1])
    ideal_worst = np.zeros(X.shape[1])

    for j in range(X.shape[1]):

        if criteria_types[j] == "benefit":
            ideal_best[j] = V[:, j].max()
            ideal_worst[j] = V[:, j].min()

        else:
            ideal_best[j] = V[:, j].min()
            ideal_worst[j] = V[:, j].max()

    distance_best = np.sqrt(
        ((V - ideal_best) ** 2).sum(axis=1)
    )

    distance_worst = np.sqrt(
        ((V - ideal_worst) ** 2).sum(axis=1)
    )

    score = distance_worst / (
        distance_best + distance_worst
    )

    return score


# -----------------------------
# AHP – jednostavna varijanta
# -----------------------------

def ahp(X, weights, criteria_types):

    scores = np.zeros_like(X, dtype=float)

    for j in range(X.shape[1]):

        column = X[:, j]

        if criteria_types[j] == "benefit":
            scores[:, j] = column / column.sum()

        else:
            inverse = 1 / column
            scores[:, j] = inverse / inverse.sum()

    final_scores = scores @ weights

    return final_scores


# -----------------------------
# VIKOR
# -----------------------------

def vikor(X, weights, criteria_types, v=0.5):

    best = np.zeros(X.shape[1])
    worst = np.zeros(X.shape[1])

    for j in range(X.shape[1]):

        if criteria_types[j] == "benefit":
            best[j] = X[:, j].max()
            worst[j] = X[:, j].min()

        else:
            best[j] = X[:, j].min()
            worst[j] = X[:, j].max()

    S = np.zeros(X.shape[0])
    R = np.zeros(X.shape[0])

    for i in range(X.shape[0]):

        values = []

        for j in range(X.shape[1]):

            if best[j] == worst[j]:
                value = 0

            elif criteria_types[j] == "benefit":
                value = weights[j] * (
                    (best[j] - X[i, j]) /
                    (best[j] - worst[j])
                )

            else:
                value = weights[j] * (
                    (X[i, j] - best[j]) /
                    (worst[j] - best[j])
                )

            values.append(value)

        S[i] = sum(values)
        R[i] = max(values)

    S_best = S.min()
    S_worst = S.max()

    R_best = R.min()
    R_worst = R.max()

    Q = (
        v * (S - S_best) / (S_worst - S_best)
        +
        (1 - v) * (R - R_best) / (R_worst - R_best)
    )

    return Q


# -----------------------------
# REZULTATI
# -----------------------------

topsis_scores = topsis(X, weights, criteria_types)
ahp_scores = ahp(X, weights, criteria_types)
vikor_scores = vikor(X, weights, criteria_types)

print("\n==============================")
print(" MCDM USPOREDBA")
print("==============================")

print("\nTOPSIS:")
for name, score in zip(alternative_names, topsis_scores):
    print(f"{name}: {score:.4f}")

print("\nAHP:")
for name, score in zip(alternative_names, ahp_scores):
    print(f"{name}: {score:.4f}")

print("\nVIKOR:")
for name, score in zip(alternative_names, vikor_scores):
    print(f"{name}: {score:.4f}")

print("\n==============================")

print(
    "\nTOPSIS najbolja:",
    alternative_names[np.argmax(topsis_scores)]
)

print(
    "AHP najbolja:",
    alternative_names[np.argmax(ahp_scores)]
)

print(
    "VIKOR najbolja:",
    alternative_names[np.argmin(vikor_scores)]
)

print("==============================")