# Keep It CALM: Analyzing the Limits of Global Unsafety in Text-to-Image Generation

<p align="center">
  <a href="https://arxiv.org/abs/2610.02300"><img src="https://img.shields.io/badge/arXiv-2610.02300-B31B1B.svg" alt="arXiv"></a>
</p>

<p align="center">
  <b>NaHyeon Park</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Minhyun Lee</b><sup>2</sup> &nbsp;·&nbsp;
  <b>Hyunjung Shim</b><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>KAIST &nbsp;&nbsp; <sup>2</sup>Samsung Electronics
</p>

> ⚠️ **Content warning.** The benchmark prompts in `data/` (and images generated from them) include
> sexually explicit, violent and otherwise offensive content. They are provided for safety research only.

## Summary

Many safeguards for text to image models remove a single global unsafe direction or subspace from every prompt. We show that this global view faces a trade off between coverage and selectivity. In other words, a compact subspace misses diverse unsafe content, while a broader one distorts benign prompts.

**CALM** (Counterfactual Adaptive Local Modulation) replaces global removal with local counterfactual correction. Using matched pairs of unsafe and benign anchors, CALM routes each prompt to the unsafe categories it actually activates. It then moves only the violating token representations, toward their benign counterparts.

![Qualitative results](assets/quali.png)

## Installation

```bash
git clone https://github.com/nahyeonkaty/calm.git
cd calm
uv sync
```

For evaluation, download the NudeNet classifier:

```bash
mkdir -p pretrained
wget -O pretrained/nudenet_classifier_model.onnx \
  "https://huggingface.co/gqfwqgw/NudeNet_classifier_model/resolve/main/classifier_model.onnx?download=true"
```

## Data

| File | Source |
|---|---|
| `data/1_nudity-ring-a-bell.csv` | [Ring-A-Bell](https://github.com/chiayi-hsu/Ring-A-Bell) (via [SAFREE](https://github.com/jaehong31/SAFREE)) |
| `data/2_unlearndiffatk.csv` | [UnlearnDiffAtk](https://github.com/OPTML-Group/Diffusion-MU-Attack) |
| `data/3_mma-diffusion-nsfw-adv-prompts.csv` | [MMA-Diffusion](https://huggingface.co/datasets/YijunYang280/MMA-Diffusion-NSFW-adv-prompts-benchmark) |
| `data/4_coco_30k_10k.csv` | [MS-COCO](https://cocodataset.org) 2014 val captions |
| `data/5_i2p_benchmark.csv` | [I2P](https://huggingface.co/datasets/AIML-TUDA/i2p) |

## Usage

```bash
# single prompt
uv run python calm.py --prompt "a photo of a person on the beach" --output-dir outputs/demo

# benchmark CSV -> outputs/ring_a_bell/all/<case>.png
uv run python calm.py --prompt-file data/1_nudity-ring-a-bell.csv --prompt-column "sensitive prompt" \
    --output-dir outputs/ring_a_bell

# evaluation (NudeNet ASR / TR) -> outputs/ring_a_bell/metric.json
uv run python evaluate.py outputs/ring_a_bell
```

See `uv run python calm.py --help` for all options.

## Citation

```bibtex
@inproceedings{park2026calm,
  title={Keep It CALM: Analyzing the Limits of Global Unsafety in Text-to-Image Generation},
  author={Park, NaHyeon and Lee, Minhyun and Shim, Hyunjung},
  booktitle={40th Conference on Neural Information Processing Systems},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
