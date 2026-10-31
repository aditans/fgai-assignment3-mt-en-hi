# EN-HI Autoregressive Neural Machine Translation

## Problem statement and objective

This project compares four autoregressive English-to-Hindi machine-translation systems on the same held-out test set:

1. A small Transformer encoder-decoder trained from scratch.
2. `google/mt5-small`, fine-tuned as an encoder-decoder.
3. A decoder-only LLaMA-family model, using a public fallback when gated access is unavailable.
4. A decoder-only Gemma-family model, using a public fallback when gated access is unavailable.

The goal is to compare translation quality, compute cost, model size, adaptation method, tokenization, and responsible-AI risks. The full experiment is implemented in [notebooks/FGAI_A3_MT_EN_HI.ipynb](notebooks/FGAI_A3_MT_EN_HI.ipynb).

## Dataset description and source link

The experiment uses the [IIT Bombay English-Hindi Parallel Corpus](https://huggingface.co/datasets/cfilt/iitb-english-hindi). The original dataset description is available from the [CFILT IIT Bombay corpus page](https://www.cfilt.iitb.ac.in/iitb_parallel/). The full corpus is not committed to this repository.

Pairs are filtered to 3–60 words per side, exact duplicates and empty/misaligned lines are removed, and pairs with a word-count ratio outside 0.5–2.0 are removed. A seed-42 shuffle is followed by train/validation/test selection. The notebook records the actual counts it obtains; if the requested 50,000/1,000/1,000 split is unavailable, the reduced split is written to the results and reported here after execution.

## Brief description of the models implemented

- **Basic Transformer:** SentencePiece BPE trained only on training text, 16k vocabulary, 3 encoder and 3 decoder layers, 256-dimensional model, four attention heads, label smoothing, Adam, and inverse-square-root warmup scheduling.
- **mT5:** `google/mt5-small`, fine-tuned with the prefix `translate English to Hindi:`.
- **LLaMA:** preferred `meta-llama/Llama-3.2-1B-Instruct`; if the gated checkpoint is unavailable, the notebook uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0` and records that fallback.
- **Gemma:** preferred `google/gemma-3-1b-it`; if gated access is unavailable, the notebook uses a public small decoder-only fallback and records the exact model identifier.

LLaMA-family and Gemma-family systems support zero-shot, three-shot, and LoRA fine-tuning paths. Decoder-only prompts use the same English/Hindi template, adapted only through the model's chat template when one is available.

## Execution steps and required dependencies

1. Open the notebook in Colab or Kaggle with a CUDA GPU. The first cell prints GPU information and library versions.
2. Upload or clone this repository.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. For gated checkpoints, accept the model licenses on Hugging Face and provide a token through the notebook runtime secret `HF_TOKEN`. Never put a token in a notebook, source file, commit, or results file.
5. Run all cells in order. The notebook creates the `results/` files and plots.
6. If a 16 GB T4 is unavailable, run the smoke-test configuration first. The notebook records skipped models and does not present skipped metrics as measured results.

The checked-in results are intentionally status-oriented until a GPU run has been completed. Blank metric cells mean “not measured”, not zero.

## Results obtained

The authoritative machine-readable result is [results/results_summary.csv](results/results_summary.csv). The initial repository snapshot contains no fabricated scores: model rows are marked `not_run_gpu_required` until the notebook is run on a compatible GPU.

| Model / method | BLEU | chrF | Training seconds | Inference seconds | Parameters | Status |
|---|---:|---:|---:|---:|---:|---|
| Basic Transformer | — | — | — | — | — | not_run_gpu_required |
| mT5-small | — | — | — | — | — | not_run_gpu_required |
| LLaMA-family zero-shot | — | — | — | — | — | not_run_gpu_required |
| LLaMA-family LoRA | — | — | — | — | — | not_run_gpu_required |
| Gemma-family zero-shot | — | — | — | — | — | not_run_gpu_required |
| Gemma-family LoRA | — | — | — | — | — | not_run_gpu_required |

After execution, replace this table with the generated table from `results/results_summary.csv`; do not copy numbers manually from an unverified run.

## Model comparison and evaluation

The unified evaluation uses the identical test set and greedy decoding with `max_new_tokens=128`. It reports sacreBLEU, chrF, wall-clock training/fine-tuning time, total and trainable parameters, inference time after warmup, milliseconds per sentence, and peak CUDA memory when available. Encoder-decoder beam-search rows are labelled separately when enabled.

Encoder-decoder models receive the source as an input sequence and generate a target sequence from a decoder cross-attending to the source. Decoder-only models place the source and target in one causal prompt, so prompt design, target-only loss masking, and prompt-echo removal matter. mT5 and the LLaMA/Gemma checkpoints benefit from large-scale pretraining, while the basic Transformer pays the full learning cost from scratch. SentencePiece subword segmentation is used for the basic model; pretrained model tokenizers are retained for the pretrained systems. LoRA reduces trainable parameters and memory at the cost of adapter capacity and an additional quantization/configuration dependency.

## Sampling-temperature analysis

The notebook evaluates a fixed 300-sentence test slice at temperatures 0.0, 0.2, 0.5, 0.7, 1.0, and 1.3, using three seeds for non-zero temperatures. It writes mean and standard deviation for BLEU and chrF to [results/temperature_analysis.csv](results/temperature_analysis.csv) and curves to `results/plots/temperature_*.png`. Temperature 0.0 is greedy and normally gives the most stable output. Moderate temperature can increase variation, while high temperature can increase omissions, unsupported additions, and garbling. These are tendencies to inspect, not claims about a model until the generated outputs are available.

## Responsible-AI discussion + mitigations

The dataset includes news, government, web, and other domains; it can contain domain, demographic, regional, and genre bias. Translation systems can default to gendered forms, flatten honorific distinctions, mistranslate cultural or religious terms, alter names and numbers, hallucinate content, or produce inappropriate handling of medical, political, violent, or self-harm content. Decoder-only models may also return explanations, refusals, or prompt echoes instead of translations.

The notebook creates side-by-side outputs for a seeded random sample and a hand-built probe set, flags number mismatches, untranslated English, empty or unusually long outputs, repeated n-grams, likely missing entities, and refusal/safety text, and writes a human-review checklist. These flags are triage signals, not proof that a Hindi translation is culturally correct or incorrect. Mitigations include human review for high-stakes use, confidence/quality estimation, named-entity and number preservation checks, domain-specific evaluation, red-team probes, documented refusal policies, data cleaning, and never using the system as an unsupervised decision-maker.

## Conclusion identifying the best-performing model

No best-performing model is claimed in this initial snapshot because no GPU experiment has been run and no scores have been fabricated. After execution, the conclusion must name the model with the highest measured BLEU/chrF and separately identify the best quality-per-compute model using the generated timing and parameter data. A quality ranking must be based only on the completed `results/` files.

## Limitations/honest notes

- The checked-in snapshot has not trained models and therefore contains no invented quality numbers.
- The local machine has an RTX 3050 with 4 GB VRAM, not the requested 16 GB T4; full pretrained-model runs should use Colab/Kaggle.
- Gated LLaMA/Gemma checkpoints require the user's own Hugging Face license acceptance and runtime token. Public fallbacks are recorded when used.
- Training from scratch and fine-tuning can take substantial time and may require reducing the training subset; all reductions are logged.
- Automated ethics flags require human review, especially for Hindi grammar, register, cultural meaning, and sensitive content.

