# Final frozen test: 2026-11-04

**Model choice:** EXP007 was selected solely because it had the highest fixed-development-set accuracy (72.80%=3640/5000), after 7 explicitly documented hypothesis-led edits. This selection occurred before the official MNIST test was run. Parameters were then frozen at commit `56a2637e8711beed86a462cf7fe5d499a338ba53` with SHA256:

`0e21cda124834f8587de187dd018a92b3d0033d874b3b5de60a30fe6129c9cdd`

The official MNIST test split consists of 10,000 independent handwritten samples. It was evaluated only **once** using `src/evaluate.py` and the frozen `experiments/exp-007-raise-class-7-bias/parameters.json`. It was **not used to edit any weights, biases, or preprocessing**.

## Result

- Development 5,000 sampled from train: **3640/5000 = 72.80%**.
- Official untouched test 10,000: **7490/10000 = 74.90%**.
- NVIDIA GeForce RTX 3060 Ti (8 GB), 615.71.09; CuPy CUDA inference.
- Output values are `X @ W.T + b`, `argmax` over 10 logits; `W` is exactly 10×196, `b` exactly 10.
- Test per-digit recall: 0=94.18%, 1=79.21%, 2=74.13%, 3=82.38%, 4=63.34%, 5=61.66%, 6=84.45%, 7=70.82%, 8=73.41%, 9=64.12%.
- `final_mnist_test.json` stores the confusion matrix, per-digit accuracy, 12 high-confidence failures and their 14×14 sample pixels, and all class logits for each sampled failure.
- `parameter_audit.json` lists exact SHA256 and changed weight/bias entry count for every experiment.

## Limitations and scientific honesty

This is not a state-of-the-art MNIST classifier. The LLM directly authored a coarse 14×14 linear weight matrix and biases, using class-conditional mean-image *visualizations* from 5,000 development samples and classification mistakes as human-readable cues. No automated parameter fitting, gradient descent, backpropagation, optimizer, or regression was run. A human-like manual editing process can itself use feedback from labels; therefore the approach is **not untrained**, merely **not gradient-trained**. The development set was inspected repeatedly and should not be treated as unbiased evaluation; the one-time held-out test figure (74.90%) is the appropriate final reported accuracy.
