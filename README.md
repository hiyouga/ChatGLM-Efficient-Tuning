# ChatGLM Efficient Tuning

![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/ChatGLM-Efficient-Tuning?style=social)
![GitHub Code License](https://img.shields.io/github/license/hiyouga/ChatGLM-Efficient-Tuning)
![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/ChatGLM-Efficient-Tuning)
![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)

Fine-tuning 🤖[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) model with 🤗[PEFT](https://github.com/huggingface/peft).

👋 Join our [WeChat](assets/wechat.jpg).

\[ English | [中文](README_zh.md) \]

## Changelog

[23/04/19] Now we support merging the weights of fine-tuned models trained by LoRA! Try `--checkpoint_dir checkpoint1,checkpoint2` argument for continually fine-tuning the models.

[23/04/18] Now we support training the quantized models using three fine-tuning methods! Try `quantization_bit` argument for training the model in 4/8 bits.

[23/04/12] Now we support training from checkpoints! Use `--checkpoint_dir` argument to specify the checkpoint model to fine-tune from.

[23/04/11] Now we support training with combined datasets! Try `--dataset dataset1,dataset2` argument for training with multiple datasets.

## Datasets

Our script now supports the following datasets:

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

## Fine-Tuning Methods

Our script now supports the following fine-tuning methods:

- [LoRA](https://arxiv.org/abs/2106.09685)
  - Fine-tuning the low-rank adapters of the model.
- [P-Tuning V2](https://github.com/THUDM/P-tuning-v2)
  - Fine-tuning the prefix encoder of the model.
- [Freeze](https://arxiv.org/abs/2012.14913)
  - Fine-tuning the MLPs in the last n blocks of the model.

## Requirement

- Python 3.10 and PyTorch 2.0.0
- 🤗Transformers, Datasets, Accelerate and PEFT (0.3.0.dev0 is required)
- protobuf, cpm_kernels, sentencepiece
- jieba, rouge_chinese, nltk

And **powerful GPUs**!

## Getting Started

### Prepare Data (optional)

Please refer to `data/example_dataset` for checking the details about the format. You can either use a single `.json` file or a [dataset loading script](https://huggingface.co/docs/datasets/dataset_script) with multiple files to create a custom dataset.

### Preparation (optional)

```bash
git clone https://github.com/hiyouga/ChatGLM-Efficient-Tuning.git
conda create -n cet python=3.10
conda activate cet
pip install -r requirements.txt
```

### Fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 python src/finetune.py \
    --do_train \
    --dataset alpaca_gpt4_zh \
    --finetuning_type lora \
    --output_dir path_to_checkpoint \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --max_train_samples 10000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --fp16
```

### Distributed Fine-tuning with Multiple GPUs
```bash
accelerate launch python src/finetune.py # arguments (same as above)
```

Note: distributed fine-tuning seems **not compatible with the LoRA method**.

### Evaluation (BLEU and ROUGE_CHINESE)

```bash
CUDA_VISIBLE_DEVICES=0 python src/finetune.py \
    --do_eval \
    --dataset alpaca_gpt4_zh \
    --checkpoint_dir path_to_checkpoint \
    --output_dir eval \
    --per_device_eval_batch_size 8 \
    --max_eval_samples 50 \
    --predict_with_generate
```

### Inference

```bash
CUDA_VISIBLE_DEVICES=0 python src/infer.py --checkpoint_dir path_to_checkpoint
```

### Deploy the Fine-tuned Model
```python
from .src import load_pretrained, ModelArguments
model_args = ModelArguments(checkpoint_dir=path_to_checkpoint_dir)
model, tokenizer = load_pretrained(model_args)
model = model.half().cuda()
# model.generate, model.chat()...
```

### Hardware Requirements

| Fine-tune method | Batch size | Mode |  GRAM  | Speed |
| ---------------- | ---------- | ---- | ------ | ----- |
| LoRA (r=8)       |     16     | FP16 |  28GB  | 8ex/s |
| LoRA (r=8)       |     8      | FP16 |  24GB  | 8ex/s |
| LoRA (r=8)       |     4      | FP16 |  20GB  | 8ex/s |
| LoRA (r=8)       |     4      | INT8 |  10GB  | 8ex/s |
| P-Tuning (p=16)  |     4      | FP16 |  20GB  | 8ex/s |
| P-Tuning (p=16)  |     4      | INT8 |  16GB  | 8ex/s |
| P-Tuning (p=16)  |     4      | INT4 |  12GB  | 8ex/s |
| Freeze (l=3)     |     4      | FP16 |  24GB  | 8ex/s |
| Freeze (l=3)     |     4      | INT8 |  12GB  | 8ex/s |

> Note: `r` is the lora rank, `p` is the number of prefix tokens, `l` is the number of trainable layers, `ex/s` is the examples per second in training. The `gradient_accumulation_steps` is set to `1`. All are evaluated on a single Tesla V100 (32G) GPU, they are approximated values and may vary in different GPUs.

## Fine-tuning ChatGLM: A Case

### Training Results

We use the whole `alpaca_gpt4_zh` dataset to fine-tune the ChatGLM model with LoRA (r=8) for one epoch, using the default hyper-parameters. The loss curve during training is presented below.

![training loss](assets/trainer_state.jpg)

### Evaluation Results

We select 100 instances in the `alpaca_gpt4_zh` dataset to evaluate the fine-tuned ChatGLM model and compute the BLEU and ROUGE scores. The results are presented below.

|   Score   | Original | FZ (l=2) | PT (p=16) | LoRA (r=8) |
| --------- | -------- | ----- | ----- | ----------------- |
| BLEU-4    |  15.75   | 16.85 | 16.06 | 17.01 (**+1.26**) |
| Rouge-1   |  34.51   | 36.62 | 34.80 | 36.77 (**+2.26**) |
| Rouge-2   |  15.11   | 17.04 | 15.32 | 16.83 (**+1.72**) |
| Rouge-l   |  26.18   | 28.17 | 26.35 | 28.86 (**+2.68**) |
| Params (%)|  /       | 4.35% | 0.06% | 0.06%             |

> FZ: freeze tuning, PT: P-Tuning V2 (we use `pre_seq_len=16` for fair comparison with LoRA), Params: the percentange of trainable parameters.

## Compared with Existing Implementations

- [THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning)
  - Official implementation of fine-tuning ChatGLM with [P-Tuning v2](https://github.com/THUDM/P-tuning-v2) on the [ADGEN](https://aclanthology.org/D19-1321.pdf) dataset.
  - Our fine-tuning script is largely depend on it. We further implement the [LoRA](https://arxiv.org/abs/2106.09685) tuning method. Additionally, we **dynamically** pad the inputs to the longest sequence in the batch instead of the maximum length, to accelerate the fine-tuning.
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

- [ ] Employing [LangChain](https://github.com/hwchase17/langchain) to easily build applications that are capable of leveraging external knowledge upon fine-tuned ChatGLM models.
- [ ] Implementing the alignment algorithms to align human preferrences.
  - [ ] [RLHF](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat)
  - [ ] [RRHF](https://github.com/GanjinZero/RRHF)
  - [ ] [RAFT](https://github.com/OptimalScale/LMFlow)
- [ ] Incorporating [Chinese datasets](https://github.com/brightmart/nlp_chinese_corpus) into the training sets.
  - [x] [BELLE](https://github.com/LianjiaTech/BELLE)
  - [ ] [pCLUE](https://github.com/CLUEbenchmark/pCLUE)
  - [ ] [CLUECorpus](https://github.com/CLUEbenchmark/CLUECorpus2020)
  - [x] [GuanacoDataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)
  - [x] [FireflyDataset](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)
- [ ] Incorporating [ChatGPT](https://openai.com/blog/chatgpt) & [GPT-4](https://openai.com/research/gpt-4) self-chat data into the training sets.
  - [ ] [Baize](https://github.com/project-baize/baize-chatbot)
  - [x] [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [x] Implementing the Freeze-Tuning and P-Tuning method.
- [x] Supporting Multi-GPUs fine-tuning. (but it does not support LoRA tuning)
- [x] Adding script for evaluation. (but it appears very slow, increasing batch size may help)
- [x] Loading from checkpoint.
- [x] Fine-tuning the quantized model.
- [ ] Writing a guidebook about how to fine-tune ChatGLM with this framework.
- [ ] Combining with state-of-the-art model editing algorithms. (*e.g. [MEND](https://arxiv.org/abs/2110.11309)*)
- [ ] Incorporating the [OpenAssistant Conversations Dataset](https://huggingface.co/datasets/OpenAssistant/oasst1) for SFT and alignment.

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).

## Citation

If this work is helpful, please cite as:

```bibtex
@Misc{chatglm-efficient-tuning,
  title = {ChatGLM Efficient Tuning},
  author = {hiyouga},
  howpublished = {\url{https://github.com/hiyouga/ChatGLM-Efficient-Tuning}},
  year = {2023}
}
```

## Acknowledgement

This repo benefits from [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B), [ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning) and [yuanzhoulvpi2017/zero_nlp](https://github.com/yuanzhoulvpi2017/zero_nlp). Thanks for their wonderful works.
