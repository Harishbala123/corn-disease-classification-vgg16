"""Loss functions.

1. Class-weighted categorical cross-entropy (both models).

2. Reinforcement-learning-inspired expected-regret loss (custom model only).
   The softmax classifier is treated as a stochastic *policy* pi(a | x) over the four diagnoses in a
   single-step decision problem (a contextual bandit): the agent sees a leaf image x, takes an action a (a
   diagnosis) and receives a reward R[y, a] from an agronomic reward matrix:
        +1 for a correct diagnosis, -1 for an ordinary error,
        -4 for the critical confusions Blight <-> Gray Leaf Spot (they call for different fungicide
           programmes, so confusing them is costly).
   The objective maximises the expected reward  J(x) = sum_a pi(a|x) R[y, a]. Because R is known and the
   action space is tiny, the expectation is computed exactly instead of being estimated from sampled actions,
   so the gradient equals the policy gradient with the expected reward as baseline:
        dJ/dz_a = pi(a|x) (R[y,a] - J(x)).
   We minimise the regret  R_max - J(x).
   Honest scope: this is NOT a full reinforcement-learning algorithm. There is no environment dynamics, no
   exploration, no sampled trajectories and no credit assignment over time. Mathematically it is
   cost-sensitive (expected-risk) learning, motivated by the reward-maximisation view of RL.

3. Supervised contrastive embedding loss (Khosla et al., NeurIPS 2020) with hard-negative weighting
   (custom model only): L2-normalised embeddings of the same class are pulled together. Pairs of
   Blight and Gray-Leaf-Spot images get `hard_negative_weight` x more weight in the softmax denominator,
   so the metric space separates exactly the classes that are confused most often.
"""
from __future__ import annotations

import keras
import numpy as np
from keras import ops


def reward_matrix(classes: list[str], critical_pair: list[str], correct=1.0, wrong=-1.0, critical=-4.0) -> np.ndarray:
    n = len(classes)
    R = np.full((n, n), wrong, np.float32)
    np.fill_diagonal(R, correct)
    i, j = classes.index(critical_pair[0]), classes.index(critical_pair[1])
    R[i, j] = R[j, i] = critical
    return R


@keras.saving.register_keras_serializable(package="corn")
class WeightedCE(keras.losses.Loss):
    def __init__(self, class_weights, name="weighted_ce", **kw):
        super().__init__(name=name, **kw)
        self.class_weights = [float(c) for c in class_weights]

    def call(self, y_true, y_pred):
        w = ops.sum(y_true * ops.convert_to_tensor(self.class_weights), axis=-1)
        p = ops.clip(y_pred, 1e-7, 1.0)
        return -w * ops.sum(y_true * ops.log(p), axis=-1)

    def get_config(self):
        return {**super().get_config(), "class_weights": self.class_weights}


@keras.saving.register_keras_serializable(package="corn")
class RewardRegretCE(WeightedCE):
    """Weighted CE + lambda * expected regret under the reward matrix (policy view of the classifier)."""

    def __init__(self, class_weights, reward, lambda_reward=1.0, name="reward_regret_ce", **kw):
        super().__init__(class_weights, name=name, **kw)
        self.reward = np.asarray(reward, np.float32).tolist()
        self.lambda_reward = float(lambda_reward)

    def call(self, y_true, y_pred):
        ce = super().call(y_true, y_pred)
        R = ops.convert_to_tensor(self.reward)
        r_rows = ops.matmul(y_true, R)                     # reward row R[y, :] for each sample
        expected_reward = ops.sum(y_pred * r_rows, axis=-1)
        regret = ops.max(r_rows, axis=-1) - expected_reward
        return ce + self.lambda_reward * regret

    def get_config(self):
        return {**super().get_config(), "reward": self.reward, "lambda_reward": self.lambda_reward}


@keras.saving.register_keras_serializable(package="corn")
class HardNegativeSupCon(keras.losses.Loss):
    """Supervised contrastive loss on L2-normalised embeddings; y_true = integer labels [batch] or [batch, 1]."""

    def __init__(self, temperature=0.1, critical_ids=(0, 2), hard_negative_weight=3.0, weight=1.0,
                 name="hard_negative_supcon", **kw):
        super().__init__(name=name, **kw)
        self.temperature = float(temperature)
        self.critical_ids = [int(c) for c in critical_ids]
        self.hard_negative_weight = float(hard_negative_weight)
        self.weight = float(weight)

    def call(self, y_true, z):
        y = ops.reshape(ops.cast(y_true, "int32"), (-1,))
        n = ops.shape(z)[0]
        sim = ops.matmul(z, ops.transpose(z)) / self.temperature
        sim = sim - ops.stop_gradient(ops.max(sim, axis=1, keepdims=True))
        eye = ops.eye(n)
        same = ops.cast(ops.equal(y[:, None], y[None, :]), "float32")
        pos = same * (1 - eye)
        a, b = self.critical_ids
        is_a, is_b = ops.cast(ops.equal(y, a), "float32"), ops.cast(ops.equal(y, b), "float32")
        crit = is_a[:, None] * is_b[None, :] + is_b[:, None] * is_a[None, :]
        w = (1 - eye) * (1 + (self.hard_negative_weight - 1) * crit)
        log_denom = ops.log(ops.sum(w * ops.exp(sim), axis=1) + 1e-9)
        n_pos = ops.sum(pos, axis=1)
        mean_log_prob_pos = ops.sum(pos * (sim - log_denom[:, None]), axis=1) / ops.maximum(n_pos, 1.0)
        loss = -mean_log_prob_pos * ops.cast(n_pos > 0, "float32")
        return self.weight * loss

    def get_config(self):
        return {**super().get_config(), "temperature": self.temperature, "critical_ids": self.critical_ids,
                "hard_negative_weight": self.hard_negative_weight, "weight": self.weight}
