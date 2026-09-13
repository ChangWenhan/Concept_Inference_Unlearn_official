"""MIA protocols for the unlearning evaluation.

fit_fr_attack / apply_fr: paper protocol — a linear SVM attack is trained on
the original model's probabilities for forget-class train (members) vs test
(non-members) samples and transferred to the candidate model; the forgetting
rate (Fr) is the fraction of member samples predicted as non-members.

simple_mia: per-sample cross-entropy loss, logistic-regression attack with
10-fold StratifiedShuffleSplit; gap = |accuracy - 0.5|.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def balance(member, nonmember, seed=42):
    n = min(len(member), len(nonmember))
    rng = np.random.RandomState(seed)
    im = rng.choice(len(member), n, replace=False) if len(member) > n else np.arange(len(member))
    inn = rng.choice(len(nonmember), n, replace=False) if len(nonmember) > n else np.arange(len(nonmember))
    return member[im], nonmember[inn]


def fit_fr_attack(original_member_probs, original_nonmember_probs, seed=42):
    pos, neg = balance(original_member_probs, original_nonmember_probs, seed)
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    scaler = StandardScaler().fit(x)
    clf = SVC(kernel="linear", random_state=seed).fit(scaler.transform(x), y)
    return {"scaler": scaler, "clf": clf}


def apply_fr(attack, member_probs):
    pred = attack["clf"].predict(attack["scaler"].transform(member_probs))
    return float(1.0 - (pred == 1).mean())


def simple_mia(member_losses, nonmember_losses, seed=42, n_splits=10):
    m, nm = balance(member_losses, nonmember_losses, seed)
    x = np.concatenate([nm, m]).reshape(-1, 1)
    y = np.array([0] * len(nm) + [1] * len(m))
    attack = LogisticRegression(max_iter=1000)
    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=seed)
    scores = cross_val_score(attack, x, y, cv=cv, scoring="accuracy")
    acc = float(scores.mean())
    return {"accuracy": acc, "gap": float(abs(0.5 - acc)), "n": int(len(m))}
