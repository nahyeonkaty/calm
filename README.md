# Keep It CALM: Analyzing the Limits of Global Unsafety in Text-to-Image Generation

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-coming%20soon-B31B1B.svg" alt="arXiv"></a>
</p>

<p align="center">
  <b>NaHyeon Park</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Minhyun Lee</b><sup>2</sup> &nbsp;·&nbsp;
  <b>Hyunjung Shim</b><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>KAIST &nbsp;&nbsp; <sup>2</sup>Samsung Electronics
</p>

> 😊 **Note (2026-10)**: CALM is accepted to **NeurIPS 2026**. Code will be released soon!

## Summary

Many safeguards for text to image models remove a single global unsafe direction or subspace from every prompt. We show that this global view faces a trade off between coverage and selectivity. In other words, a compact subspace misses diverse unsafe content, while a broader one distorts benign prompts.

**CALM** (Counterfactual Adaptive Local Modulation) replaces global removal with local counterfactual correction. Using matched pairs of unsafe and benign anchors, CALM routes each prompt to the unsafe categories it actually activates. It then moves only the violating token representations, toward their benign counterparts.

![Qualitative results](assets/quali.png)

## Installation

Requires Python >= 3.10. I personally recommend [uv](https://github.com/astral-sh/uv) for environment management.

```bash
git clone https://github.com/nahyeonkaty/calm.git
cd calm
pip install -e .
```

## Citation

```bibtex
@article{park2026calm,
  title={Keep It CALM: Analyzing the Limits of Global Unsafety in Text-to-Image Generation},
  author={Park, NaHyeon and Lee, Minhyun and Shim, Hyunjung},
  booktitle={40th Conference on Neural Information Processing Systems},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
