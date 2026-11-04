# MNIST Direct Weight Editing Lab

MNISTの10値手書き数字分類に対して、LLMが**重み行列とバイアスの値を明示して直接編集**する実験の記録です。

- 勾配降下法、逆伝播、Adam、SGD、最小二乗推定、訓練データからのプロトタイプ自動作成を**一切使用しません**。
- 変更ごとに独立した`experiments/exp-XXX/`ディレクトリと、GitHubのPRを作成して、検証後に`main`へマージします。
- すべての実験で、実際の`weights`（10×196）と`biases`（10）を`parameters.json`に数値配列として完全保存します。
- 重み/バイアスの変更箇所、修正根拠、実験の正解率、confusion matrixをPRで公開します。
- MNIST公式IDXの画像を28×28→14×14に2×2平均プーリングした196入力に対し、`argmax(X W^T + b)`で推論します。GPUはRTX 3060 Ti、CuPyによるCUDA行列乗算のみを使用します。
- 調整には訓練画像のうち固定seedで選んだ5,000枚を使用。**公式テスト画像10,000枚は最後の評価にだけ使用**します。

## How to run

```bash
python3 -m venv --system-site-packages venv
venv/bin/pip install 'cupy-cuda13x[ctk]'
# Obtain IDX .gz files from https://storage.googleapis.com/cvdf-datasets/mnist/
venv/bin/python src/evaluate.py experiments/exp-001-handwritten-baseline/parameters.json --split dev
```

`data/` and `venv/` are local-only and ignored by git.

## Pull request protocol

Each PR should contain: an immutable copy of the full model, metrics, a named change list with literal entries edited, the reasoning before observing the new result, and a merge decision. Changing an already-merged experiment's file is prohibited. Parameter edits are **LLM-authored numerical choices**, never produced by an optimizer.

## Experiment log and results (2026-11-04)

| PR | Edit | Dev correct/5000 | Dev accuracy |
|---|---|---:|---:|
| [#1](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/1) | Handwritten binary prototype weights, literal biases | 1130 | 22.60% |
| [#2](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/2) | Bias-only manual reallocation | 720 | 14.40% |
| [#3](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/3) | Manually redraw all 1960 weights using visual stroke masks | 3282 | 65.64% |
| [#4](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/4) | Increase b[4], b[5], b[9] by 1.5 | 3535 | 70.70% |
| [#5](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/5) | 33 explicit coordinate-wise weight edits | 3555 | 71.10% |
| [#6](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/6) | Undo 10 selected weight edits | 3554 | 71.08% |
| [#7](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/7) | Raise b[7] from -4.2 to -3.2 | **3640** | **72.80%** |

**Final (frozen EXP007) official 10,000-image test accuracy: 74.90% (7490/10000)**.

All model experiments were first reviewed as GitHub pull requests, then merged into the single repository. The final test was run once only after the best checkpoint was selected on the development set; see [`validation/README.md`](validation/README.md) and [`validation/final_mnist_test.json`](validation/final_mnist_test.json).
