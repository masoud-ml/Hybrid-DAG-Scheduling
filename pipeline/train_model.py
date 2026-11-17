"""Balanced Random Forest self-training.

An inner split selects the number of retraining cycles, followed by a
refit on the full training partition. The outer 20% is reserved for
final evaluation."""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split, learning_curve
from data.feature.features import FEATURE_COLUMNS
from learning.model import RFModelConfig, build_default_model

@dataclass(frozen=True)
class SemiSupTrainConfig:
    labels_csv: str = 'out/labels.csv'
    out_dir: str = 'out/learning'
    label_col: str = 'label'
    n_labeled_per_class: int = 40_000
    n_undef: int = 80_000
    n_unseen: int = 20_000
    test_size: float = 0.20
    split_seed: int = 123
    max_cycles: int = 5
    confidence_threshold: float = 0.90
    min_improve: float = 1e-4
    model_name: str = 'rf'
    rf_cfg: RFModelConfig = field(default_factory=RFModelConfig)
    make_plots: bool = True
    strict_counts: bool = False


def _normalize_label_with_undefined(s):
    x = pd.to_numeric(s, errors='coerce')
    bad = x.notna() & ~x.isin([0, 1])
    if bad.any():
        raise ValueError('Invalid labels: expected 0, 1, or undefined')
    return x


def _paper_feature_cols_for_your_csv(df):
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'Missing paper features: {sorted(missing)}')
    return list(FEATURE_COLUMNS)


def _metrics(model, X, y):
    pred = model.predict(X)
    return dict(acc=float(accuracy_score(y, pred)),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                classification_report=classification_report(y, pred, labels=[0, 1],
                    target_names=['anomaly_free', 'anomaly'], zero_division=0, output_dict=True))


def _new_model(cfg):
    return build_default_model(model_name=cfg.model_name, rf_cfg=cfg.rf_cfg)


def fit_self_training(X, y, pool, cfg, *, monitor=None, cycles=None):
    """Fit self-training rounds, optionally selecting cycles on an inner split."""
    model = _new_model(cfg).fit(X, y)
    rounds = [(model, X.copy(), y.copy())]
    history = [dict(cycle=0, n_labeled_train=len(y), added=0, remaining=len(pool))]
    best_score = None
    if monitor is not None:
        best_score = accuracy_score(monitor[1], model.predict(monitor[0]))
        history[0]['inner_accuracy'] = float(best_score)
    best_cycle = 0
    for cycle in range(1, (cfg.max_cycles if cycles is None else cycles) + 1):
        if not len(pool):
            break
        proba = model.predict_proba(pool)
        take = np.max(proba, axis=1) > cfg.confidence_threshold
        if not take.any():
            break
        labels = model.sklearn_model.classes_[proba[take].argmax(axis=1)]
        X = np.concatenate([X, pool[take]])
        y = np.concatenate([y, labels])
        pool = pool[~take]
        model = _new_model(cfg).fit(X, y)
        rounds.append((model, X.copy(), y.copy()))
        item = dict(cycle=cycle, n_labeled_train=len(y), added=int(take.sum()), remaining=len(pool))
        history.append(item)
        if monitor is not None:
            score = accuracy_score(monitor[1], model.predict(monitor[0]))
            item['inner_accuracy'] = float(score)
            if score <= best_score + cfg.min_improve:
                break
            best_score = score
        best_cycle = cycle
    return rounds, history, best_cycle


def _plots(out, rounds, Xva, yva, unseen, cols, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for cycle, (model, X, y) in enumerate(rounds):
        if min(np.sum(y == 0), np.sum(y == 1)) >= 5:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            sizes, tr, te = learning_curve(model.sklearn_model, X, y, cv=cv,
                train_sizes=np.linspace(.25, 1, 4), n_jobs=1, scoring='accuracy')
            pd.DataFrame({'n': sizes, 'train': tr.mean(axis=1), 'cv': te.mean(axis=1)}).to_csv(
                out / f'fig7_round{cycle}_learning_curve.csv', index=False)
            plt.figure()
            plt.plot(sizes, tr.mean(axis=1), 'o-', label='Training')
            plt.plot(sizes, te.mean(axis=1), 'o-', label='5-fold CV')
            plt.xlabel('Training examples'); plt.ylabel('Accuracy'); plt.legend()
            plt.tight_layout(); plt.savefig(out / f'fig7_round{cycle}.png', dpi=150); plt.close()
        if len(unseen):
            prob = model.predict_proba(unseen)
            pred = model.sklearn_model.classes_[prob.argmax(axis=1)]
            confidence = prob.max(axis=1)
            plt.figure()
            for cls in (0, 1):
                plt.hist(confidence[pred == cls], bins=np.linspace(.5, 1, 11), alpha=.6,
                         label=f'Predicted {cls}')
            plt.xlabel('Prediction confidence (unseen scenarios)'); plt.ylabel('Count')
            plt.legend(); plt.tight_layout()
            plt.savefig(out / f'fig10_round{cycle}.png', dpi=150); plt.close()
    model = rounds[-1][0]
    cm = confusion_matrix(yva, model.predict(Xva), labels=[0, 1])
    plt.figure(); plt.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2): plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.xticks([0, 1]); plt.yticks([0, 1]); plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout(); plt.savefig(out / 'fig8_confusion_matrix.png', dpi=150); plt.close()
    importance = pd.Series(model.sklearn_model.feature_importances_, index=cols).sort_values()
    plt.figure(figsize=(8, 5)); importance.plot.barh()
    plt.xlabel('Random Forest impurity-based importance'); plt.tight_layout()
    plt.savefig(out / 'fig9_feature_importance.png', dpi=150); plt.close()


def main(cfg=None):
    cfg = cfg or SemiSupTrainConfig()
    if not 0 < cfg.test_size < 1 or not 0 <= cfg.confidence_threshold < 1 or cfg.max_cycles < 0:
        raise ValueError('Invalid training configuration')
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg.labels_csv)
    cols = _paper_feature_cols_for_your_csv(df)
    if 'dag_id' in df and df.dag_id.duplicated().any():
        raise ValueError('Repeated DAG IDs can leak across partitions; provide independent scenarios')
    features = df[cols].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError('Features contain NaN/inf')
    target = _normalize_label_with_undefined(df[cfg.label_col])
    rng = np.random.default_rng(cfg.split_seed)
    classes = [np.flatnonzero(target.to_numpy() == cls) for cls in (0, 1)]
    n = min(cfg.n_labeled_per_class, *(len(c) for c in classes))
    if n < 5:
        raise ValueError('Need >=5 independent labeled examples per class for nested splitting')
    if cfg.strict_counts and n != cfg.n_labeled_per_class:
        raise ValueError('Paper mode requires the requested labeled count in each class')
    chosen = np.concatenate([rng.choice(c, n, replace=False) for c in classes])
    tr, va = train_test_split(chosen, test_size=cfg.test_size, random_state=cfg.split_seed,
                              stratify=None)
    unknown = np.flatnonzero(target.isna())
    if cfg.strict_counts and len(unknown) < cfg.n_undef:
        raise ValueError('Insufficient undefined pool for paper protocol')
    unknown = rng.choice(unknown, min(cfg.n_undef, len(unknown)), replace=False)
    remaining = np.setdiff1d(np.arange(len(df)), np.concatenate([chosen, unknown]))
    if cfg.strict_counts and len(remaining) < cfg.n_unseen:
        raise ValueError('Need 20,000 separate unseen scenarios in paper mode')
    unseen_idx = rng.choice(remaining, min(cfg.n_unseen, len(remaining)), replace=False)
    X = features.to_numpy(dtype=float)
    y = target.to_numpy()
    fit, monitor = train_test_split(tr, test_size=.2, random_state=cfg.split_seed + 1,
                                    stratify=y[tr])
    _, selection_history, selected = fit_self_training(X[fit], y[fit].astype(int), X[unknown], cfg,
                                                      monitor=(X[monitor], y[monitor].astype(int)))
    rounds, history, _ = fit_self_training(X[tr], y[tr].astype(int), X[unknown], cfg, cycles=selected)
    final_model = rounds[-1][0]
    final = _metrics(final_model, X[va], y[va].astype(int))
    partition = np.full(len(df), 'unused', dtype=object)
    for name, ix in [('training', tr), ('validation', va), ('undefined_training', unknown),
                     ('unseen', unseen_idx)]: partition[ix] = name
    pd.DataFrame({'row': np.arange(len(df)), 'dag_id': df.get('dag_id', pd.Series(df.index)),
                  'partition': partition}).to_csv(out / 'split_manifest.csv', index=False)
    audit = dict(selection_history=selection_history, selected_cycles=selected,
                 replay_history=history, validation_used_for_training=False,
                 counts={k: int(np.sum(partition == k)) for k in np.unique(partition)},
                 stopping_policy='inner 20% of training, then replay on full training',
                 learning_curve_note='Augmented-round CV uses pseudo-labels generated before CV; diagnostic, not unbiased ground-truth accuracy.')
    (out / 'training_history.json').write_text(json.dumps(audit, indent=2))
    (out / 'classification_report.json').write_text(json.dumps(final, indent=2))
    (out / 'classification_report.txt').write_text(classification_report(y[va], final_model.predict(X[va]), zero_division=0))
    pd.DataFrame(final['confusion_matrix'], index=['actual_0','actual_1'], columns=['pred_0','pred_1']).to_csv(out / 'confusion_matrix.csv')
    pd.Series(final_model.sklearn_model.feature_importances_, index=cols, name='importance').to_csv(out / 'feature_importance.csv')
    if len(unseen_idx):
        predictions = pd.DataFrame({'row': unseen_idx, 'dag_id': df.iloc[unseen_idx]['dag_id'].to_numpy() if 'dag_id' in df else unseen_idx})
        for cycle, (model, _, _) in enumerate(rounds):
            probs = model.predict_proba(X[unseen_idx])
            predictions[f'round{cycle}_label'] = model.sklearn_model.classes_[probs.argmax(axis=1)]
            predictions[f'round{cycle}_confidence'] = probs.max(axis=1)
        predictions.to_csv(out / 'unseen_predictions.csv', index=False)
    if cfg.make_plots:
        _plots(out, rounds, X[va], y[va], X[unseen_idx], cols, cfg.split_seed)
    for cycle, (model, _, _) in enumerate(rounds):
        joblib.dump({'model': model, 'feature_cols': cols, 'train_config': asdict(cfg),
                     'training_protocol': 'paper-inspired; documented inner stopping split'},
                     out / f'round{cycle}_model.joblib')
    joblib.dump({'model': final_model, 'feature_cols': cols, 'train_config': asdict(cfg),
                'validation_accuracy': final['acc'], 'training_protocol': 'isolated-final-holdout'},
                out / 'trained_model.joblib')
    print(json.dumps({'balanced_per_class': n, 'validation_count': len(va), 'unseen_count': len(unseen_idx),
                      'accuracy': final['acc'], 'selected_cycles': selected}))
    return final

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', default='out/labels.csv'); p.add_argument('--out', default='out/learning')
    p.add_argument('--per-class', type=int, default=40000); p.add_argument('--undefined', type=int, default=80000)
    p.add_argument('--unseen', type=int, default=20000); p.add_argument('--trees', type=int, default=100)
    p.add_argument('--max-cycles', type=int, default=5); p.add_argument('--no-plots', action='store_true')
    p.add_argument('--strict-counts', action='store_true')
    a = p.parse_args()
    main(SemiSupTrainConfig(labels_csv=a.labels, out_dir=a.out, n_labeled_per_class=a.per_class,
        n_undef=a.undefined, n_unseen=a.unseen, max_cycles=a.max_cycles, strict_counts=a.strict_counts,
        rf_cfg=RFModelConfig(n_estimators=a.trees), make_plots=not a.no_plots))
