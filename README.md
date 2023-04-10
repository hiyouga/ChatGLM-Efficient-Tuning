# ChatGLM Efficient Tuning

![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/ChatGLM-Efficient-Tuning?style=social)
![GitHub Code License](https://img.shields.io/github/license/hiyouga/ChatGLM-Efficient-Tuning)
![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/ChatGLM-Efficient-Tuning)


Fine-tuning 🤖[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) model with 🤗[PEFT](https://github.com/huggingface/peft).


## Datasets

Now our script supports the following datasets:

- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [Stanford Alpaca (Chinese)](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [GPT-4 Generated Data](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [BELLE 2M](https://huggingface.co/datasets/BelleGroup/train_2M_CN)
- [BELLE 1M](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
- [BELLE 0.5M](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
- [BELLE Dialogue 0.4M](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)
- [BELLE School Math 0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M)
- [BELLE Multiturn Chat 0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)
- [Guanaco Dataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)
- [Firefly 1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)

Please refer to `config_data.py` for details.

## Requirement

- Python 3.10 and PyTorch 2.0.0
- 🤗Transformers, Datasets, and PEFT
- protobuf, cpm_kernels, sentencepiece

And **powerful GPUs**!

## Getting Started

### Preparation

```bash
git clone https://github.com/hiyouga/ChatGLM-Efficient-Tuning.git
conda create -n cet python=3.10
conda activate cet
pip install -r requirements.txt
```

### Fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_chatglm.py \
    --do_train \
    --dataset guanaco \
    --output_dir output_guanaco \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --warmup_steps 100 \
    --max_train_samples 10000 \
    --learning_rate 5e-4 \
    --num_train_epochs 1.0 \
    --fp16
```

### Inference

```bash
CUDA_VISIBLE_DEVICES=0 python infer_chatglm.py
```

### Hardware Requirements

| Batch size | LoRA `r` | Mode | GRAM |
| ---------- | -------- | ---- | ---- |
|     8      |     8    | FP16 | 24GB |


## Compared with Existing Implementations
- [THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning)
  - Official implementation of fine-tuning ChatGLM with [P-Tuning v2](https://github.com/THUDM/P-tuning-v2) on the [ADGEN](https://aclanthology.org/D19-1321.pdf) dataset.
  - Our fine-tuning script is largely depend on it. We further implement the [LoRA](https://arxiv.org/abs/2106.09685) tuning method. Additionally, we **dynamically** pad the inputs to the longest sequence in the batch instead of the maximum length.
- [mymusise/ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning)
  - An unoffical implementation of fine-tuning ChatGLM with [LoRA](https://arxiv.org/abs/2106.09685) on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset.
  - We borrowed some ideas from it. Our fine-tuning script **integrates** the data pre-processing part into the training procedure, so we need not generate a pre-processed dataset before training.
- [ssbuild/chatglm_finetuning](https://github.com/ssbuild/chatglm_finetuning)
  - An unofficial implementation of fine-tuning ChatGLM with several PEFT methods on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset.
  - Our fine-tuning script is implemented **purely** with [Huggingface transformers](https://github.com/huggingface/transformers) and is independent of the [deep_training](https://github.com/ssbuild/deep_training) framework.
- [lich99/ChatGLM-finetune-LoRA](https://github.com/lich99/ChatGLM-finetune-LoRA)
  - An unofficial implementation of fine-tuning ChatGLM with [LoRA](https://arxiv.org/abs/2106.09685) on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset.
  - We use the [Huggingface PEFT](https://github.com/huggingface/peft) to provide the state-of-the-art PEFT methods.
- [liucongg/ChatGLM-Finetuning](https://github.com/liucongg/ChatGLM-Finetuning)
  - An unofficial implementation of fine-tuning ChatGLM with several methods including Freeze, LoRA and P-Tuning on the industrial dataset.
  - We are aim to incorporate more instruction-following datasets for fine-tuning the ChatGLM model.
- [yanqiangmiffy/InstructGLM](https://github.com/yanqiangmiffy/InstructGLM)
  - An unofficial implementation of fine-tuning ChatGLM that explores the ChatGLM's ability on the instruction-following datasets.
  - Our fine-tuning script integrates the data pre-processing part in to the training procedure.


## TODO

- Incorporating [Chinese datasets](https://github.com/brightmart/nlp_chinese_corpus) into the training sets.
  - [BELLE](https://github.com/LianjiaTech/BELLE)
  - [pCLUE](https://github.com/CLUEbenchmark/pCLUE)
  - [CLUECorpus](https://github.com/CLUEbenchmark/CLUECorpus2020)
  - ~~[GuanacoDataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)~~
  - ~~[FireflyDataset](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)~~
- Incorporating [ChatGPT](https://openai.com/blog/chatgpt) & [GPT-4](https://openai.com/research/gpt-4) self-chat data into the training sets.
  - [Baize](https://github.com/project-baize/baize-chatbot)
  - ~~[GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)~~
- Implementing the Freeze-Tuning and P-Tuning method.
- Supporting Multi-GPUs fine-tuning.


## License

This repository is licensed under the [Apache-2.0 License](LICENSE).


## Citation

If this work is helpful, please cite as:

```bibtex
@Misc{cet,
  title = {ChatGLM Efficient Tuning},
  author = {hiyouga},
  howpublished = {\url{https://github.com/huggingface/peft}},
  year = {2023}
}
```


## Acknowledgement

This repo benefits from [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B), [ChatGLM-Tuning](https://github.com/THUDM/ChatGLM-6B) and [yuanzhoulvpi2017/zero_nlp](https://github.com/yuanzhoulvpi2017/zero_nlp). Thanks for their wonderful works.
