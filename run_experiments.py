"""Sequential experiment runner. External CUDA_VISIBLE_DEVICES selects the GPU."""

import argparse
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/fgsrcs"))
    p.add_argument("--output", type=Path, default=Path("runs"))
    p.add_argument("--seeds", type=int, nargs="+", default=[27042, 27043, 27044])
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--ablations", action="store_true")
    p.add_argument("--weights", type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    common = ["--data", str(a.data.resolve()), "--output", str(a.output.resolve())]

    def run(script, extra):
        subprocess.run(
            [sys.executable, str(root / script), *common, *extra], check=True
        )

    for seed in a.seeds:
        for fold in range(4):
            args = ["--seed", str(seed), "--fold", str(fold)]
            dest = a.output / f"b{seed}" / f"fold{fold}"
            if not (dest / "training_complete.json").exists():
                run(
                    "train_backbone.py",
                    args
                    + (["--weights", str(a.weights.resolve())] if a.weights else []),
                )
            if not (dest / "features/complete.json").exists():
                run("extract_features.py", args)
            if not (dest / "method/complete.json").exists():
                run(
                    "score_methods.py",
                    args + (["--controls"] if a.ablations and seed == 27042 else []),
                )
            if a.baselines and not (dest / "posthoc/complete.json").exists():
                run("score_baselines.py", args)
            if a.ablations and seed == 27042:
                for ratio in [0.5, 0.25]:
                    for poolseed in [23001, 23002]:
                        if not (
                            dest / f"pool{ratio}_seed{poolseed}/complete.json"
                        ).exists():
                            run(
                                "score_methods.py",
                                args
                                + [
                                    "--pool-ratio",
                                    str(ratio),
                                    "--pool-seed",
                                    str(poolseed),
                                ],
                            )
    run("evaluate.py", ["--seeds", *map(str, a.seeds)])
    if a.baselines:
        run("evaluate.py", ["--setting", "posthoc", "--seeds", *map(str, a.seeds)])
    if a.ablations and 27042 in a.seeds:
        for ratio in [0.5, 0.25]:
            for ps in [23001, 23002]:
                run(
                    "evaluate.py",
                    ["--setting", f"pool{ratio}_seed{ps}", "--seeds", "27042"],
                )
