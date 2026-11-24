#!/usr/bin/env python3
"""Python 3 port of the PhonemeDiscovery scoring pipeline.

The legacy tree is retained as the historical reference. This port preserves
its inventory, segmentation, aggregation, TF-IDF/LSI, and pair-sampling logic,
while making paths and experimental parameters explicit.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from gensim import corpora, matutils, models
from gensim.similarities import MatrixSimilarity

LEVELS = {
    "Eng": ["h_2", "h_4", "h_10", "h_19", "phonemic", "htk_80", "htk_160", "htk_320", "htk_640"],
    "Jap": ["h_2", "h_4", "h_8", "h_13", "phonemic", "htk_50", "htk_100", "htk_200", "htk_400"],
}
CONTEXTS = (5, 10, 20, 50, 100, 200, 500)


def read_lines(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8") as stream:
        return [line.split() for line in stream]


def derive_segmentation(unsegmented: list[list[str]], coarse: list[list[str]], source: Path) -> list[list[str]]:
    """Apply each coarse utterance's colon-defined boundaries to finer units."""
    if len(unsegmented) != len(coarse):
        raise ValueError(f"utterance count mismatch: {source} has {len(coarse)}, input has {len(unsegmented)}")
    output: list[list[str]] = []
    for line_no, (fine, coarse_line) in enumerate(zip(unsegmented, coarse), start=1):
        chunks: list[list[str]] = []
        for word in coarse_line:
            width = len(word.split(":"))
            chunks.append(fine[:width])
            fine = fine[width:]
        if fine:
            raise ValueError(f"{source}:{line_no}: boundary lengths do not consume the finer utterance")
        output.append([":".join(chunk) for chunk in chunks])
    return output


def aggregate(utterances: list[list[str]], window: int) -> list[list[str]]:
    """Match agregate.awk: concatenate window-sized groups of utterances."""
    result = []
    for start in range(0, len(utterances), window):
        result.append([token for row in utterances[start : start + window] for token in row])
    return result


def make_word_map(coarse: list[list[str]], fine: list[list[str]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    coarse_tokens = [token for row in coarse for token in row]
    fine_tokens = [token for row in fine for token in row]
    if len(coarse_tokens) != len(fine_tokens):
        raise ValueError("coarse and fine token streams have different lengths")
    for parent, child in zip(coarse_tokens, fine_tokens):
        children = mapping.setdefault(parent, [])
        if child not in children:
            children.append(child)
    return mapping


def score(corpus_rows: list[list[str]], word_map: dict[str, list[str]], *, dim: int,
          repetitions: int, rng: random.Random) -> tuple[float, float]:
    dictionary = corpora.Dictionary(corpus_rows)
    singleton_ids = [token_id for token_id, frequency in dictionary.dfs.items() if frequency == 1]
    dictionary.filter_tokens(singleton_ids)
    dictionary.compactify()
    if len(dictionary) < 2:
        raise ValueError("fewer than two vocabulary items remain after singleton filtering")

    bow = [dictionary.doc2bow(row) for row in corpus_rows]
    tfidf = models.TfidfModel(bow)
    lsi = models.LsiModel(tfidf[bow], id2word=dictionary, num_topics=dim)
    # Retain gensim's historical term-vector construction (U multiplied by S).
    term_vectors = np.multiply(lsi.projection.u, lsi.projection.s)
    term_corpus = matutils.Dense2Corpus(term_vectors.T)
    index = MatrixSimilarity(term_corpus)
    terms = list(term_corpus)
    id_to_token = {value: key for key, value in dictionary.token2id.items()}
    score_ids = [dictionary.token2id[token] for children in word_map.values() for token in children
                 if token in dictionary.token2id]
    if len(score_ids) < 2:
        raise ValueError("fewer than two mapped sub-word tokens occur in the filtered vocabulary")

    aucs = []
    for _ in range(repetitions):
        labeled_scores = []
        for children in word_map.values():
            present = [token for token in children if token in dictionary.token2id]
            if len(present) <= 1:
                continue
            token_a, token_b = rng.sample(present, 2)
            id_a, id_b = dictionary.token2id[token_a], dictionary.token2id[token_b]
            labeled_scores.append((1, float(index[terms[id_a]][id_b])))
            noise_a, noise_b = rng.sample(range(len(id_to_token)), 2)
            labeled_scores.append((0, float(index[terms[noise_a]][noise_b])))
        aucs.append(binary_auc(labeled_scores))
    return float(np.mean(aucs)), float(np.std(aucs))


def binary_auc(data: list[tuple[int, float]]) -> float:
    positives = sum(label for label, _ in data)
    negatives = len(data) - positives
    if not positives or not negatives:
        raise ValueError("AUC needs at least one positive and one negative pair")
    ordered = sorted((value, label) for label, value in data)
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        average_rank = ((i + 1) + j) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def run(args: argparse.Namespace) -> None:
    root = args.repo_root.resolve()
    legacy = root / "legacy" / "PhonemeDiscovery"
    corpus_root = args.corpus_dir.resolve() if args.corpus_dir else legacy / "corpus"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    segmentation = args.segmentation
    language = args.language
    levels = LEVELS[language]
    rows_out = []

    for level_index in range(len(levels) - 1):
        coarse_name, fine_name = levels[level_index : level_index + 2]
        coarse_path = corpus_root / f"corpus_{segmentation}" / f"{coarse_name}.{language}"
        fine_gold_path = corpus_root / "corpus_gold" / f"{fine_name}.{language}"
        coarse_rows = read_lines(coarse_path)
        fine_input = [[unit for token in row for unit in token.split(":")] for row in read_lines(fine_gold_path)]
        fine_rows = derive_segmentation(fine_input, coarse_rows, coarse_path)
        word_map = make_word_map(coarse_rows, fine_rows)

        for context in args.contexts:
            agg_rows = aggregate(fine_rows, context)
            mean_auc, std_auc = score(agg_rows, word_map, dim=args.dim,
                                      repetitions=args.repetitions, rng=rng)
            row = (segmentation, language, coarse_name, context, args.dim, mean_auc, std_auc)
            rows_out.append(row)
            print("\t".join(map(str, row)))
    with output.open("w", encoding="utf-8") as stream:
        stream.write("Segmentation\tlanguage\tvariation\tcontext\tdimension\tAUC\tStD\n")
        for row in rows_out:
            stream.write("\t".join(map(str, row)) + "\n")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=LEVELS)
    parser.add_argument("segmentation", choices=("gold", "seg", "rand"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--corpus-dir", type=Path, help="corpus directory (defaults to legacy/PhonemeDiscovery/corpus)")
    parser.add_argument("--output", type=Path, default=Path("res/modern-results.tsv"))
    parser.add_argument("--contexts", type=int, nargs="+", choices=CONTEXTS, default=[20])
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.dim < 1 or args.repetitions < 1:
        parser.error("--dim and --repetitions must be positive")
    try:
        run(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
