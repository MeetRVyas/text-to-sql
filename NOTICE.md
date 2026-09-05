# NOTICE

This project fine-tunes and redistributes derivatives of, and depends on,
third-party models and datasets that carry their own license terms,
separate from this project's own MIT license (see `LICENSE`).

## Qwen2.5-Coder-3B-Instruct (base model)

This project's default base model, `Qwen/Qwen2.5-Coder-3B-Instruct`, is
licensed under the **Qwen RESEARCH LICENSE AGREEMENT** (Tongyi Qianwen
Research License) — **not** Apache 2.0. This differs from most other sizes
in the same Qwen2.5-Coder family (0.5B / 1.5B / 7B / 14B / 32B are
Apache 2.0; only the 3B and 72B sizes use the research license). Confirmed
directly against the model's Hugging Face model card.

Key terms (see the full agreement for the authoritative text — this is a
summary, not legal advice):
- **Non-commercial use only.** Commercial use requires a separate license
  from Alibaba Cloud.
- If you redistribute the Materials — which includes the LoRA adapter
  trained in this project, as a derivative work built on top of the base
  model — you must include a copy of the license agreement, mark any files
  you modified, and retain the attribution notice below.

Required attribution notice, per the license's own redistribution terms:

> Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.

Full agreement:
https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/blob/main/LICENSE

This project (training code, evaluation code, API, and the public demo) is
personal and non-commercial, consistent with these terms. If you fork this
project for a commercial use case, you'll need to either switch to one of
the Apache-2.0-licensed Qwen2.5-Coder sizes (0.5B/1.5B/7B/14B/32B) or
request a commercial license from Alibaba Cloud for the 3B model.

## Spider dataset

Training and evaluation data (`scripts/download_spider.sh`, and the curated
demo databases in `space/`) come from the Spider text-to-SQL dataset,
released by Yale under a non-commercial research license:
https://yale-lily.github.io/spider

If you use Spider, its authors ask that you cite:

Yu, Tao, et al. "Spider: A Large-Scale Human-Labeled Dataset for Complex
and Cross-Domain Semantic Parsing and Text-to-SQL Task." EMNLP 2018.

## This project's own code

Everything else in this repository (training/evaluation/API code, this
NOTICE file, etc.) is released under the MIT license — see `LICENSE`.