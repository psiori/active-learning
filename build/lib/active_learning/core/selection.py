"""Explicit orchestration helpers for active-learning runs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from active_learning.core.image_provider import ImageProvider
from active_learning.scorers.alges import score_alges_embeddings
from active_learning.scorers.brightness import filter_by_brightness
from active_learning.scorers.features import score_features
from active_learning.scorers.uncertainty import (
    score_bald,
    score_entropy,
    score_mc_dropout,
)
from active_learning.selectors.alges import select_alges
from active_learning.selectors.coreset import select_coreset
from active_learning.selectors.uncertainty import (
    select_uncertainty_coreset,
    select_uncertainty_topk,
    select_uncertainty_topk_then_coreset,
)
from active_learning.core.types import SelectionResult

logger = logging.getLogger(__name__)

SelectionProgressSink = Callable[[int, int, str | None], None]


class SelectionProgress:
    """Maps phased (done/total) work into monotonic percent 0..99 with API total 100."""

    __slots__ = ("_sink", "_last")

    def __init__(self, sink: SelectionProgressSink | None):
        self._sink = sink
        self._last = 0

    def span(self, start: int, width: int) -> SelectionProgressSpan:
        return SelectionProgressSpan(self, start, width)

    def emit(self, start: int, width: int, done: int, total: int, message: str) -> None:
        if self._sink is None:
            return
        if total <= 0:
            inner = width if done > 0 else 0
        else:
            inner = int(width * min(done, total) / total)
        pct = min(99, max(self._last, start + inner))
        self._last = pct
        self._sink(pct, 100, message)


class SelectionProgressSpan:
    __slots__ = ("_root", "_start", "_width")

    def __init__(self, root: SelectionProgress, start: int, width: int):
        self._root = root
        self._start = start
        self._width = width

    def subspan(self, t0: float, t1: float) -> SelectionProgressSpan:
        a = self._start + int(self._width * t0)
        b = self._start + int(self._width * t1)
        w = max(1, b - a)
        return SelectionProgressSpan(self._root, a, w)

    def step(self, done: int, total: int, message: str) -> None:
        self._root.emit(self._start, self._width, done, total, message)


def build_image_provider(
    provider_source,
    cache_root: str | Path,
    *,
    retained_metadata_fields: tuple[str, ...] | list[str] = (),
    ignore_missing_blobs: bool = False,
) -> ImageProvider:
    return ImageProvider(
        provider_source,
        cache_root=cache_root,
        retained_metadata_fields=retained_metadata_fields,
        ignore_missing_blobs=ignore_missing_blobs,
    )


def apply_brightness_predicate(
    sample_ids: list[str],
    image_provider: ImageProvider,
    cache_root: str | Path,
    *,
    min_brightness: float = 0.0,
    max_brightness: float = 220.0,
    progress: bool = False,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    n_in = len(sample_ids)
    logger.info("Filtering by brightness...")
    filtered, stats = filter_by_brightness(
        sample_ids,
        image_provider,
        cache_root,
        min_brightness=min_brightness,
        max_brightness=max_brightness,
        progress=progress,
    )
    n_out = len(filtered)
    logger.info("Brightness filtering kept %d/%d images.", n_out, n_in)
    return filtered, stats


def run_coreset_selection(
    candidate_ids: list[str],
    *,
    seed_ids: list[str],
    image_provider: ImageProvider,
    cache_root: str | Path,
    n: int,
    feature_model: str = "resnet50",
    seed: int | None = None,
    progress: bool = False,
    progress_scope: SelectionProgressSpan | None = None,
) -> SelectionResult:
    n_candidates = len(candidate_ids)
    logger.info("Running coreset selection (image features + diversity)...")
    if progress_scope is not None:
        if seed_ids:
            span_cand = progress_scope.subspan(0.0, 0.55)
            span_seed = progress_scope.subspan(0.55, 0.93)
            span_tail = progress_scope.subspan(0.93, 1.0)
        else:
            span_cand = progress_scope.subspan(0.0, 0.94)
            span_seed = None
            span_tail = progress_scope.subspan(0.94, 1.0)
    else:
        span_cand = span_seed = span_tail = None

    def _feat_step(span: SelectionProgressSpan | None, label: str):
        if span is None:
            return None
        return lambda d, t: span.step(d, t, f"{label} ({d}/{t})")

    features = score_features(
        candidate_ids,
        image_provider,
        cache_root,
        model_name=feature_model,
        progress=progress,
        on_step=_feat_step(span_cand, "Candidate features"),
    )
    artifacts = {"features": features}
    if seed_ids:
        artifacts["seed_features"] = score_features(
            seed_ids,
            image_provider,
            cache_root,
            model_name=feature_model,
            progress=progress,
            on_step=_feat_step(span_seed, "Seed features"),
        )
    if span_tail is not None:
        span_tail.step(1, 1, "Choosing diverse subset")
    result = select_coreset(
        candidate_ids,
        n=n,
        artifacts=artifacts,
        seed_ids=seed_ids,
        seed=seed,
    )
    logger.info(
        "Coreset selection chose %d/%d images.",
        len(result.selected_ids),
        n_candidates,
    )
    return result


def run_uncertainty_selection(
    candidate_ids: list[str],
    *,
    seed_ids: list[str],
    image_provider: ImageProvider,
    cache_root: str | Path,
    n: int,
    uncertainty_kind: str,
    uncertainty_model,
    feature_model: str = "resnet50",
    model_name: str | None = None,
    model_path: str | None = None,
    alpha: float = 0.5,
    candidate_multiplier: int = 4,
    iterations: int = 5,
    image_size: tuple[int, int] = (320, 240),
    batch_size: int = 8,
    aggregation: str = "topk_mean",
    topk_fraction: float = 0.10,
    strategy: str = "uncertainty_coreset",
    seed: int | None = None,
    progress: bool = False,
    progress_scope: SelectionProgressSpan | None = None,
) -> SelectionResult:
    n_candidates = len(candidate_ids)
    logger.info(
        f"Running uncertainty selection (strategy={strategy!r}, "
        f"uncertainty={uncertainty_kind!r})..."
    )

    def _u_on(sp: SelectionProgressSpan | None):
        if sp is None:
            return None
        return lambda d, t: sp.step(d, t, f"Uncertainty ({d}/{t})")

    def _f_on(sp: SelectionProgressSpan | None, label: str):
        if sp is None:
            return None
        return lambda d, t: sp.step(d, t, f"{label} ({d}/{t})")

    u_sp = fc_sp = fs_sp = tail_sp = None
    if progress_scope is not None:
        if strategy == "uncertainty_topk":
            u_sp = progress_scope.subspan(0.0, 0.92)
            tail_sp = progress_scope.subspan(0.92, 1.0)
        elif strategy == "uncertainty_topk_coreset":
            if seed_ids:
                u_sp = progress_scope.subspan(0.0, 0.26)
                fc_sp = progress_scope.subspan(0.26, 0.52)
                fs_sp = progress_scope.subspan(0.52, 0.68)
                tail_sp = progress_scope.subspan(0.68, 1.0)
            else:
                u_sp = progress_scope.subspan(0.0, 0.28)
                fc_sp = progress_scope.subspan(0.28, 0.62)
                tail_sp = progress_scope.subspan(0.62, 1.0)
        else:
            if seed_ids:
                u_sp = progress_scope.subspan(0.0, 0.30)
                fc_sp = progress_scope.subspan(0.30, 0.65)
                fs_sp = progress_scope.subspan(0.65, 0.82)
                tail_sp = progress_scope.subspan(0.82, 1.0)
            else:
                u_sp = progress_scope.subspan(0.0, 0.33)
                fc_sp = progress_scope.subspan(0.33, 0.88)
                tail_sp = progress_scope.subspan(0.88, 1.0)

    u_on = _u_on(u_sp)
    if uncertainty_kind == "bald":
        uncertainty = score_bald(
            candidate_ids,
            image_provider,
            cache_root,
            mc_unet=uncertainty_model,
            model_name=model_name,
            model_path=model_path,
            iterations=iterations,
            image_size=image_size,
            batch_size=batch_size,
            aggregation=aggregation,
            topk_fraction=topk_fraction,
            progress=progress,
            on_step=u_on,
        )
    elif uncertainty_kind == "mc_dropout":
        uncertainty = score_mc_dropout(
            candidate_ids,
            image_provider,
            cache_root,
            mc_unet=uncertainty_model,
            model_name=model_name,
            model_path=model_path,
            iterations=iterations,
            image_size=image_size,
            batch_size=batch_size,
            aggregation=aggregation,
            topk_fraction=topk_fraction,
            progress=progress,
            on_step=u_on,
        )
    else:
        uncertainty = score_entropy(
            candidate_ids,
            image_provider,
            cache_root,
            unet=uncertainty_model,
            model_name=model_name,
            model_path=model_path,
            image_size=image_size,
            batch_size=batch_size,
            aggregation=aggregation,
            topk_fraction=topk_fraction,
            progress=progress,
            on_step=u_on,
        )

    artifacts = {"uncertainty": uncertainty}
    if strategy != "uncertainty_topk":
        artifacts["features"] = score_features(
            candidate_ids,
            image_provider,
            cache_root,
            model_name=feature_model,
            progress=progress,
            on_step=_f_on(fc_sp, "Candidate features"),
        )
        if seed_ids:
            artifacts["seed_features"] = score_features(
                seed_ids,
                image_provider,
                cache_root,
                model_name=feature_model,
                progress=progress,
                on_step=_f_on(fs_sp, "Seed features"),
            )

    if tail_sp is not None:
        tail_sp.step(1, 1, "Applying selector")

    if strategy == "uncertainty_topk":
        result = select_uncertainty_topk(candidate_ids, n=n, artifacts=artifacts)
    elif strategy == "uncertainty_topk_coreset":
        result = select_uncertainty_topk_then_coreset(
            candidate_ids,
            n=n,
            artifacts=artifacts,
            seed_ids=seed_ids,
            candidate_multiplier=candidate_multiplier,
            seed=seed,
        )
    else:
        result = select_uncertainty_coreset(
            candidate_ids,
            n=n,
            artifacts=artifacts,
            seed_ids=seed_ids,
            alpha=alpha,
            seed=seed,
        )
    logger.info(
        "Uncertainty selection chose %d/%d images.",
        len(result.selected_ids),
        n_candidates,
    )
    return result


def run_alges_selection(
    candidate_ids: list[str],
    *,
    image_provider: ImageProvider,
    cache_root: str | Path,
    model,
    model_name: str | None = None,
    model_path: str | None = None,
    n: int,
    method: str = "semantic",
    image_size: tuple[int, int] = (320, 240),
    batch_size: int = 8,
    seed: int | None = None,
    progress: bool = False,
    progress_scope: SelectionProgressSpan | None = None,
) -> SelectionResult:
    n_candidates = len(candidate_ids)
    logger.info("Running ALGES selection (embeddings + selector)...")
    emb_sp = tail_sp = None
    if progress_scope is not None:
        emb_sp = progress_scope.subspan(0.0, 0.92)
        tail_sp = progress_scope.subspan(0.92, 1.0)

    def _emb_on(sp: SelectionProgressSpan | None):
        if sp is None:
            return None
        return lambda d, t: sp.step(d, t, f"ALGES inference ({d}/{t})")

    embeddings = score_alges_embeddings(
        candidate_ids,
        image_provider,
        cache_root,
        model=model,
        model_name=model_name,
        model_path=model_path,
        image_size=image_size,
        method=method,
        batch_size=batch_size,
        progress=progress,
        on_step=_emb_on(emb_sp),
    )
    if tail_sp is not None:
        tail_sp.step(1, 1, "Running k-means++ selection")
    result = select_alges(
        candidate_ids,
        n=n,
        artifacts={"alges_embeddings": embeddings},
        seed=seed,
        progress=progress,
    )
    logger.info(
        "ALGES selection chose %d/%d images.",
        len(result.selected_ids),
        n_candidates,
    )
    return result


def run_selection(
    cfg,
    *,
    candidate_ids: list[str],
    seed_ids: list[str],
    image_provider: ImageProvider,
    uncertainty_model=None,
    alges_model=None,
    progress: bool = False,
    selection_progress: SelectionProgressSink | None = None,
) -> SelectionResult:
    prog = SelectionProgress(selection_progress) if selection_progress else None
    scope = prog.span(0, 99) if prog else None

    strategy = cfg.selection.strategy
    if strategy == "coreset":
        return run_coreset_selection(
            candidate_ids,
            seed_ids=seed_ids,
            image_provider=image_provider,
            cache_root=cfg.query.cache_root,
            n=cfg.selection.n_select,
            feature_model=cfg.coreset.feature_model,
            seed=cfg.selection.seed,
            progress=progress,
            progress_scope=scope,
        )

    if strategy in (
        "uncertainty_coreset",
        "uncertainty_topk",
        "uncertainty_topk_coreset",
    ):
        uc = cfg.uncertainty_coreset
        model_def = cfg.models[uc.uncertainty_model]
        return run_uncertainty_selection(
            candidate_ids,
            seed_ids=seed_ids,
            image_provider=image_provider,
            cache_root=cfg.query.cache_root,
            n=cfg.selection.n_select,
            uncertainty_kind=uc.provider,
            uncertainty_model=uncertainty_model,
            feature_model=uc.feature_model,
            model_name=model_def.name,
            model_path=model_def.path,
            alpha=uc.alpha,
            candidate_multiplier=uc.candidate_multiplier,
            iterations=uc.mc_iterations,
            image_size=model_def.image_size,
            batch_size=uc.batch_size,
            aggregation=uc.aggregation,
            topk_fraction=uc.topk_fraction,
            strategy=strategy,
            seed=cfg.selection.seed,
            progress=progress,
            progress_scope=scope,
        )

    al = cfg.alges
    model_def = cfg.models[al.model]
    if strategy == "alges":
        return run_alges_selection(
            candidate_ids,
            image_provider=image_provider,
            cache_root=cfg.query.cache_root,
            model=alges_model,
            model_name=model_def.name,
            model_path=model_def.path,
            n=cfg.selection.n_select,
            method=al.method,
            image_size=model_def.image_size,
            batch_size=al.batch_size,
            seed=cfg.selection.seed,
            progress=progress,
            progress_scope=scope,
        )

    from active_learning.strategies.alges_coreset import run_alges_coreset_selection

    return run_alges_coreset_selection(
        candidate_ids,
        seed_ids=seed_ids,
        image_provider=image_provider,
        cache_root=cfg.query.cache_root,
        model=alges_model,
        model_name=model_def.name,
        model_path=model_def.path,
        feature_model=cfg.coreset.feature_model,
        n=cfg.selection.n_select,
        method=al.method,
        image_size=model_def.image_size,
        batch_size=al.batch_size,
        seed=cfg.selection.seed,
        progress=progress,
        selection_progress=selection_progress,
    )
