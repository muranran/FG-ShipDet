# FG-ShipDet

Code for **Fine-Grained Open-Set Ship Recognition by Integrating Sample Distances and Neighborhood Information**.

This release contains the FGSRCS public-dataset experiments: object-crop preparation, known-label ResNet50 training, feature extraction, relative-distance regression, single-step and multi-step neighborhood fusion, post-hoc baselines, and evaluation. The task is recognition of annotated ship crops, rather than full-image object detection.

## Installation

Use Python 3.10 or 3.11 on Linux and a CUDA-capable GPU for backbone training. Install a compatible PyTorch/torchvision pair for your GPU first. The experiment environment used PyTorch 2.7.1 and torchvision 0.22.1 with CUDA 12.6.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

The 18-method baseline suite additionally requires libmr for OpenMax. A C/C++ compiler is required if a binary wheel is unavailable.

```bash
python -m pip install Cython
python -m pip install libmr==0.1.9
```

## Dataset

Obtain FGSRCS from its [official repository](https://github.com/dwddw/FGSRCS), following the dataset authors' access and usage conditions. Images and pretrained weights are not bundled here. The expected source structure is:

```text
FGSRCS/
  train/
    images/*.png
    annfiles_lv2/*.txt
  val/
    images/*.png
    annfiles_lv2/*.txt
```

Prepare all 11,187 level-2 objects from the 4,500 train/validation images:

```bash
python prepare_fgsrcs.py --source /path/to/FGSRCS --output data/fgsrcs
```

The script crops each polygon's axis-aligned bounding rectangle and builds four class partitions. Each of the 17 classes is held out once: the four partitions contain 13/4, 13/4, 13/4, and 12/5 known/unknown classes. Class partition seed: 27142. Known calibration images are selected using a deterministic source-image hash. Calibration normalization and decision subsets are separated by source image.

The official validation images form the independent test set. Test objects are excluded from training and the reference pool by path, source image, and pixel hashes. Pool manifests contain no category labels. The reference pool contains other images from unknown classes; it does not contain the test images. Test queries access a fixed reference graph without becoming reference nodes or propagating to one another.

## Main experiments

```bash
# Three backbones x four class partitions; three distance heads per combination.
python run_experiments.py --data data/fgsrcs --output runs

# Include all 18 post-hoc baselines and the fixed ablations.
python run_experiments.py --data data/fgsrcs --output runs --baselines --ablations
```

The runner is sequential. Select a device using your local `CUDA_VISIBLE_DEVICES` setting. To schedule individual combinations separately:

```bash
python train_backbone.py --data data/fgsrcs --output runs --fold 0 --seed 27042
python extract_features.py --data data/fgsrcs --output runs --fold 0 --seed 27042
python score_methods.py --data data/fgsrcs --output runs --fold 0 --seed 27042
python score_baselines.py --data data/fgsrcs --output runs --fold 0 --seed 27042
```

Repeat for folds 0–3 and backbone seeds 27042, 27043, 27044. The scoring script trains heads with seeds 24042, 24043, 24044. The backbone uses known labeled training images; its parameters are frozen before reference-pool processing. The 5120–256–1 distance head uses known training and unlabeled reference features. Backbone training and distance-head training each use a fixed 30 epochs; head training uses 49 updates per epoch.

The default neighborhood size is 15, propagation coefficient is 0.9, and distance/neighborhood scores are independently normalized using the known normalization subset before equal-weight fusion. Local features are three PCA-axis strips from ResNet50 layer3; global features are from its final pooled layer. The existing `max(R,G,B)>10` support heuristic is retained in local feature extraction.

For offline ImageNet initialization, supply `--weights /path/to/resnet50-11ad3fa6.pth` to the runner or backbone script. Otherwise torchvision downloads its ImageNet1K V2 weights.

## Evaluation

```bash
python evaluate.py --data data/fgsrcs --output runs
python evaluate.py --data data/fgsrcs --output runs --setting posthoc
```

Evaluation requires frozen scores and thresholds for all four partitions and all requested backbone seeds. For a complete four-partition single-seed run, add `--seeds 27042`. Test labels are read after the scoring barrier. Thresholds use the known decision subset's higher quantile at 1%, 5%, and 10% calibration budgets; rejection is strictly greater than the threshold. Actual test false rejection is reported separately.

Outputs include:

- Unknown rejection rate averaged equally over the 17 held-out classes.
- Actual known false-rejection rate.
- Known correct-acceptance rate averaged over known classes within each partition.
- H-score: harmonic mean of unknown rejection and known correct acceptance.
- Per-partition/head records and mean/sample standard deviation across independent backbone seeds.

The FGSRCS reference results below use the 5% calibration setting. They are results of the original experiments, not a fresh end-to-end run of this reorganized release.

| Method | Unknown rejection (%) | Known false rejection (%) | Known correct acceptance (%) | H (%) |
|---|---:|---:|---:|---:|
| Distance regression | 51.0407 | 5.1543 | 75.2704 | 60.8098 |
| Regression + single-step neighborhood | 53.4132 | 5.1000 | 76.3720 | 62.8386 |
| Regression + multi-step neighborhood | 54.4171 | 5.2970 | 75.5426 | 63.2431 |
| Raw relative distance + single-step neighborhood | 52.6432 | 5.0656 | 76.6877 | 62.4214 |
| Raw relative distance + multi-step neighborhood | 53.7373 | 5.2264 | 75.9217 | 62.9229 |

## Baselines and ablations

`score_baselines.py` implements MSP, entropy, energy, maximum logit, cosine prototype distance, empirical and shrinkage Mahalanobis distance, KNN (1/10/50/1000), residual/ViM, ReAct (90/99 percentiles), and three OpenMax configurations. OpenMax uses correctly classified known training logits, libmr tail fitting, and rank weighting. The configurations are controlled adaptations to the same backbone and crop protocol.

`score_methods.py --controls` adds energy fusion, degree-preserving random graph rewiring (three seeds), neighborhood sizes 5/30, and propagation coefficients 0.5/0.7. These controls use the first backbone seed in the original experiments.

`--pool-ratio 0.5 --pool-seed 23001` (or 0.25, seed 23002) retains nested source-image groups from the eligible unlabeled reference pool, keeps labeled training images fixed, and retrains all three distance heads. The ratio applies to unlabeled source groups, not the entire labeled-plus-unlabeled reference set. Evaluate a completed four-partition pool setting with, for example:

```bash
python evaluate.py --data data/fgsrcs --output runs --setting pool0.5_seed23001 --seeds 27042
```

## Files

| File | Purpose |
|---|---|
| `prepare_fgsrcs.py` | Public dataset crops and independent-query manifests |
| `train_backbone.py` | Known-label backbone fitting |
| `extract_features.py`, `fgship_features.py` | Frozen global/local feature extraction |
| `score_methods.py`, `fgship_core.py` | Distance targets, heads, graph scores, fusion and controls |
| `score_baselines.py`, `fgship_openmax.py` | Post-hoc comparisons |
| `evaluate.py` | Frozen-score evaluation and aggregation |
| `run_experiments.py` | Sequential orchestration |
| `test_core.py` | CPU unit checks (`python -m unittest test_core -v`) |

Completed stages are not overwritten. Keep experiments in separate output directories when changing settings. Runtime data, weights, feature caches and credentials are excluded by `.gitignore`.

## Acknowledgments

Please cite the FGSRCS dataset and the original methods when using their data or comparisons. See [FGSRCS](https://github.com/dwddw/FGSRCS), [OpenMax](https://github.com/abhijitbendale/OSDN), and [libMR](https://github.com/Vastlab/libMR). Third-party datasets, weights, and packages remain subject to their respective terms.
