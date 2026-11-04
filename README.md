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
