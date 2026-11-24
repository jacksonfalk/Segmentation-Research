# Python 3 pipeline

This is a Python 3 port of the scoring path in `legacy/PhonemeDiscovery`. The
legacy directory remains the unchanged historical snapshot. The modern runner
uses Python 3 and Gensim 4, and does not need GNU sed or awk.

## Install

Use Python 3.10 or newer. On Narval, load a compatible Python module available
to your account, then make a virtual environment and install the pinned
dependency. For example, after checking versions with `module spider python`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r modern/requirements.txt
```

Keep the environment in a suitable project or home location according to your
Alliance allocation guidance; do not put it in scratch if scratch is purged.

## Run

From the repository root:

```bash
python modern/run.py Eng gold --contexts 20 --dim 200 --seed 0 \
  --output res/modern-Eng-gold.tsv
```

Use `Jap` or segmentation `seg` / `rand` as needed. `--contexts` accepts one or
more of 5, 10, 20, 50, 100, 200, and 500. The default reproduces the legacy
runner's single context size (20), dimension (200), and 100 sampling
repetitions. The seed is explicit so a run is repeatable. `--corpus-dir` can
point to a separately stored copy of the input corpus directory.

The script builds the derived segmentation and context windows in memory, then
writes a TSV result table. It reports each completed inventory comparison to
stdout as it runs. It expects the repository's existing corpus files and
retains the legacy score definition: pairwise cosine similarity of randomly
selected sub-word pairs versus random vocabulary pairs, evaluated with AUC.

## Port notes

- Python 2 syntax and dictionary iterator APIs were replaced with Python 3.
- The Japanese branch typo is avoided by using one shared language-level
  table for both languages.
- GNU `sed` and `awk` preprocessing was moved into Python.
- The AUC calculation handles score ties by average ranks; this may yield tiny
  numerical differences from the bundled legacy `pyroc.py` implementation.
- This ports the repository's downstream scoring stage. It does not recreate
  the upstream audio, MFCC, HTK, DiBS, or Adaptor Grammar steps described in
  the paper.

For a defensible comparison, record the Python, NumPy, SciPy, and Gensim
versions, cluster/module names, command line, and random seed with each run.
