from __future__ import annotations

import numpy as np

from active_learning.providers import bald, entropy, mc_dropout


class TensorLike:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def numpy(self) -> np.ndarray:
        return self.value


def _probs() -> np.ndarray:
    return np.array(
        [
            [
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.4, 0.6], [0.7, 0.3]],
            ]
        ],
        dtype=np.float32,
    )


def test_entropy_scores_can_focus_on_target_classes():
    def infer(_batch):
        return {"probs": TensorLike(_probs())}

    whole = entropy.entropy_scores_for_batch(infer, object(), aggregation="mean")
    focused = entropy.entropy_scores_for_batch(
        infer,
        object(),
        aggregation="mean",
        target_classes=[1],
    )

    entropy_map = entropy.entropy_map(_probs())
    target_mask = np.array([[[False, True], [True, False]]])
    np.testing.assert_allclose(whole, np.array([entropy_map.mean()], dtype=np.float32))
    np.testing.assert_allclose(
        focused,
        np.array([entropy_map[target_mask].mean()], dtype=np.float32),
    )


def test_bald_scores_can_focus_on_target_classes(monkeypatch):
    mc_probs = np.stack([_probs(), _probs()], axis=0)
    monkeypatch.setattr(bald, "collect_mc_probs", lambda *_args: mc_probs)

    score = bald.bald_scores_for_batch(
        lambda _batch: None,
        object(),
        2,
        aggregation="mean",
        target_classes=[0],
    )

    np.testing.assert_allclose(score, np.array([0.0], dtype=np.float32))


def test_mc_dropout_scores_return_zero_when_no_target_pixels(monkeypatch):
    mc_probs = np.stack([_probs(), _probs()], axis=0)
    monkeypatch.setattr(mc_dropout, "collect_mc_probs", lambda *_args: mc_probs)

    score = mc_dropout.mc_dropout_scores_for_batch(
        lambda _batch: None,
        object(),
        2,
        aggregation="mean",
        target_classes=[99],
    )

    np.testing.assert_allclose(score, np.array([0.0], dtype=np.float32))
