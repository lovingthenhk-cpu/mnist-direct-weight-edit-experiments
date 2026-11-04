# MNIST手書き文字10分類・99%達成実験レポート

実験日：2026年11月4日  
環境：RTX 3060 Ti（8GB）、NVIDIAドライバ615.71.09、CUDA、CuPy 14.2.0、NumPy 2.5.4、SciPy 1.18.1  
記録：[mnist-direct-weight-edit-experiments](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments)

## 要旨

本実験は、**勾配降下法・逆伝播・自動的な重み学習を使用せず**、LLMが数値係数とバイアスを直接指定するMNIST手書き数字10クラス分類を検証したものである。

最初に全1,960重みと10バイアスをLLMが直接作成した線形分類器を試し、最高公式テスト精度**74.90%**を得た。次に、ラベル付き訓練画像を保持する参照検索型の分類方式に移行し、手動で設定した固定の画像変換・形状距離を比較した。

最終モデル（実験015）は開発用5,000枚で**99.04%（4,952枚正解）**、パラメータを凍結した後の公式テスト10,000枚で**99.02%（9,902枚正解、98枚誤分類）**だった。観測上の99%目標を達成した。ただし、後半の方式は55,000〜60,000枚のラベル付き参照画像を用いるため、「分類に利用するすべての情報をLLMが一から記述した」という条件を満たすものではない。

## 1. 目的と制約

目標はMNISTの10クラス分類で99%以上の正解率を観測すること。SGD、Adam、逆伝播、勾配降下、自動パラメータ推定は行わず、係数の値をLLMが事前に文章化・明示設定する。実験ごとにGitHubの新規ブランチでモデルの数値と結果を保存し、PRとしてmainにマージする。精度が下がった結果も削除しない。

**二つの方式を厳密に区別した**。実験001〜007の線形モデルは全係数がLLM記述の数値配列である。実験009〜015は参照用訓練画像そのものに分類に必要なラベル情報が保持される非パラメトリック方式であり、手動で作成したのは距離計算係数・バイアス等の明示的な数値である。

## 2. MNISTデータと分割

公式MNISTは28×28画素の訓練画像60,000枚、テスト画像10,000枚。画素は255で割って0〜1に正規化した。固定乱数シード20261104で訓練画像を並べ替え、先頭5,000枚を開発用とし、残り55,000枚だけを参照画像にした。**開発用5,000枚が自分自身を最近傍として検索することはない**。

最終評価のときだけ参照画像を全訓練60,000枚とし、公式テスト10,000枚を推論した。参照する画像を増やすことは標準的な訓練データの利用であり、係数の再学習ではない。

注意：本シリーズより前の旧線形モデルで公式テストの集計情報が閲覧済みであった。したがって**プロジェクト全体として完全に未知のテストではない**が、新モデルの公式テストは実験015の数値を確定・固定した後に初めて実施した。実験009〜015の調整には公式テストを用いていない。

## 3. モデルの仕組み

### 3.1 全重みを直接指定した線形モデル（旧方式）

画像を14×14へ平均プーリングし、10×196個の重みWと10個のバイアスbをLLMが明示した。クラスcのスコアは、196入力の内積とバイアスの和である。この方式では数値係数を全て追跡できるが、最高モデルの公式テストは7,490/10,000=74.90%であった。

### 3.2 参照画像検索と手動指定の画像変形距離（新方式）

まず画像の二次モーメントに基づく**固定の傾斜補正**を行い、標準偏差0.65のガウス平滑化を適用する。ユークリッド距離で最も近い80枚の参照画像を選び、それぞれに**局所的な±1画素のずれ**を許した距離を計算する。

各入力画素pと参照画素の局所9候補を比較するときのコスト：

\[
  (I_p-R_{p+\Delta})^2
  +2.5(G^x_p-G^x_{p+\Delta})^2
  +2.5(G^y_p-G^y_{p+\Delta})^2
  +0.004(\Delta x^2+\Delta y^2).
\]

各画素で最小コストを選び全画素について足した値を局所距離とする。最終距離は

\[
  d=0.12\,d_{\mathrm{pixel}}+0.88\,d_{\mathrm{local}}.
\]

最短3近傍のラベルへ、各重み1/(sqrt(d)+0.1)を加点して最大クラスを選択する。**10個のクラスバイアスはいずれも0.0**であり、数値はJSONに全文保存した。

ここで使う Gx, Gy は画像の**空間的な有限差分**であり、損失関数を重みについて微分した「学習の勾配」とは違う。**勾配降下は実施していない**。

## 4. 実験と結果

正解率は断りのない限り**開発用5,000枚**に対する値。

| 実験 | PR | 内容 | 正解数 | 正解率 |
|---|---|---|---:|---:|
| 001 | [#1](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/1) | 初期の完全手書き重み | 1,130 | 22.60% |
| 002 | [#2](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/2) | バイアス全体を手動更新 | 720 | 14.40% |
| 003 | [#3](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/3) | 全1,960重みを再設計 | 3,282 | 65.64% |
| 004 | [#4](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/4) | バイアス3個を更新 | 3,535 | 70.70% |
| 005 | [#5](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/5) | 画素の重み33個を更新 | 3,555 | 71.10% |
| 006 | [#6](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/6) | 重み10個を戻す | 3,554 | 71.08% |
| 007 | [#7](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/7) | バイアス1個を手動更新 | 3,640 | 72.80% |
| 009 | [#9](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/9) | 生の28×28の3近傍法 | 4,868 | 97.36% |
| 010 | [#10](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/10) | 幾何学的傾斜補正 | 4,898 | 97.96% |
| 011 | [#11](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/11) | 局所形状距離 | 4,926 | 98.52% |
| 012 | [#12](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/12) | 候補80→256に拡大 | 4,925 | 98.50% |
| 013 | [#13](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/13) | 固定ガウス平滑化 | 4,928 | 98.56% |
| 014 | [#14](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/14) | 局所ずれを±2に拡張 | 4,921 | 98.42% |
| **015** | [#15](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments/pull/15) | **方向の空間微分係数0.35→2.5（2箇所）** | **4,952** | **99.04%** |

実験008は旧方式の評価監査PRであり、モデルの実験番号ではない。

## 5. 最終公式テスト

| 対象 | 正解数/総数 | 正解率 |
|---|---:|---:|
| 全重みを手動で作成した旧線形モデル（EXP007） | 7,490 / 10,000 | 74.90% |
| 新方式の開発用（参照55,000枚） | 4,952 / 5,000 | 99.04% |
| **新方式の公式テスト（参照60,000枚）** | **9,902 / 10,000** | **99.02%** |

正解率は旧線形モデルから**24.12ポイント**改善。ただし異なるモデル構造・参照画像の利用を伴うため、単純に手動重み編集だけの効果とは結論できない。

### 数字別再現率

| 数字 | 正解数 / 実数 | 再現率 |
|---:|---:|---:|
| 0 | 976 / 980 | 99.59% |
| 1 | 1,129 / 1,135 | 99.47% |
| 2 | 1,020 / 1,032 | 98.84% |
| 3 | 998 / 1,010 | 98.81% |
| 4 | 965 / 982 | 98.27% |
| 5 | 881 / 892 | 98.77% |
| 6 | 952 / 958 | 99.37% |
| 7 | 1,018 / 1,028 | 99.03% |
| 8 | 966 / 974 | 99.18% |
| 9 | 997 / 1,009 | 98.81% |

間違いは**98枚**。数字4を9と誤分類したものが11枚、9を4と誤分類したものが7枚あり、筆跡形状の近い数字の区別に限界が残る。全混同行列と誤認例は [公式テスト結果JSON](validation/exp015-99percent/results_official_test.json) に収録した。

10,000件を独立な二項標本と仮定したWilsonの95%信頼区間は**約98.81〜99.20%**。今回のテストで観測された正解率は99%を超えたが、真の母集団精度が99%以上であることを統計学的に確定したわけではない。

## 6. 凍結・記録・再現

最終仕様を選定後、公式テスト前のGitコミットは

1fbffeccc71f5dbc46acf29ef66feea9fd281734

パラメータファイルSHA256は

36de40657e828f841c62f38665dc165a9db8c79b4bbbe46448fbe20c6108baef

である。係数は [EXP015の完全なパラメータJSON](experiments/exp-015-edge-oriented-manual-metric/parameters.json) に保存してあり、推論実装は [局所画像変形距離](src/local_distortion_manual.py) と [最近傍検索・前処理](src/nearest_manual.py) を参照する。

再現にはMNIST公式IDX圧縮ファイルをdataフォルダに置き、CUDA対応CuPy、NumPy、SciPyを利用する。

~~~bash
python3 -m venv --system-site-packages venv
venv/bin/pip install 'cupy-cuda13x[ctk]' scipy numpy

venv/bin/python src/local_distortion_manual.py \
  experiments/exp-015-edge-oriented-manual-metric/parameters.json --split dev

venv/bin/python src/local_distortion_manual.py \
  experiments/exp-015-edge-oriented-manual-metric/parameters.json --split test
~~~

これらのコマンドは重みや距離係数を更新しない。テストセットは再現のために実行できるが、モデルの再調整には使っていない。

## 7. 限界と検証上の注意

**第一に、全情報をLLMが書いたモデルで99%に到達したわけではない。** ラベル付き訓練データは参照ライブラリという記憶として残り、分類に不可欠である。厳密に重みを一から手動指定した旧モデルの最高値は公式テスト74.90%である。

**第二に、固定の開発用5,000枚を複数回見て係数を選んでいる。** 勾配降下を使わないからといって、開発セットへの人間的な過適合が起こらないわけではない。公式テストはEXP015の係数を確定後に一度だけ実行したが、前シリーズのテスト集計情報が既知だったためプロジェクト全体の完全ブラインド試験ではない。

**第三に、推論時に大量の参照画像を保持・検索する。** 計算量・保存容量・移植性は小規模なニューラルネットワークより不利な場合がある。今回はRTX 3060 Tiを利用した10,000枚の実測が約6.44秒だったが、他の環境の性能保証ではない。

## 8. 結論

**最終観測結果は、MNIST公式テスト99.02%（9,902/10,000枚）であり、99%の目標を達成した。** 勾配降下法・逆伝播を使わず、距離係数2個を人手で0.35から2.5に変更して最後の改善を得た。ただし大量のラベル付き参照画像を利用する方式であり、純粋な手書きの全重みを持つモデルではないという制約を明記して報告する。

## 参考資料

- [MNISTデータセット](https://yann.lecun.com/exdb/mnist/) / [公式データのGoogleホスト版](https://storage.googleapis.com/cvdf-datasets/mnist/)
- [KNNを用いた手書き数字(MNIST)の分類とエラー分析](https://zenn.dev/kumamoto/articles/7edd14cd368bf6)
- [Matching Shapes：形状比較と変形](https://vision.cornell.edu/se3/wp-content/uploads/2014/09/00937552.pdf)
- [MNISTにおける近傍法・変形距離の歴史的比較](https://www.ians.uni-stuttgart.de/anm/publications/files_publication_anm/Ha05_www_Dissertation_Uni_Freiburg.pdf)
- [全PR、モデル、実験記録のGitHubリポジトリ](https://github.com/lovingthenhk-cpu/mnist-direct-weight-edit-experiments)
