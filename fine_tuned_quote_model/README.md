---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:3731
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Fate is like a strange, unpopular restaurant filled with odd little
    waiters who bring you things you never asked for and don't always like. [AUTHOR]
    Lemony Snicket [TAGS] humor simile fate
  sentences:
  - And still, after all this time,The sun never says to the earth,You owe Me.Look
    what happens withA love like that,It lights the Whole Sky. [AUTHOR] Hafiz [TAGS]
    inspiration faith
  - A word after a word after a word is power. [AUTHOR] Margaret Atwood [TAGS] writing
    reading
  - I have not failed. I've just found 10,000 ways that won't work. [AUTHOR] Thomas
    A. Edison [TAGS] inspirational failure paraphrased edison
- source_sentence: In heaven, all the interesting people are missing. [AUTHOR] Friedrich
    Nietzsche [TAGS] heaven religion
  sentences:
  - Don't Gain The World  Lose Your Soul, Wisdom Is Better Than Silver Or Gold. [AUTHOR]
    Bob Marley [TAGS] soul peace wealth wisdom
  - He who has a why to live for can bear almost any how. [AUTHOR] Friedrich Nietzsche
    [TAGS] purpose questioning how why questions life
  - If you're lonely when you're alone, you're in bad company. [AUTHOR] Jean-Paul
    Sartre [TAGS] loneliness solitude
- source_sentence: You have brains in your head. You have feet in your shoes. You
    can steer yourself any direction you choose. You're on your own. And you know
    what you know. And YOU are the one who'll decide where to go... [AUTHOR] Dr. Seuss,  [TAGS]
    inspirational
  sentences:
  - The more that you read, the more things you will know. The more that you learn,
    the more places you'll go. [AUTHOR] Dr. Seuss,  [TAGS] seuss learning reading
  - The most effective way to destroy people is to deny and obliterate their own understanding
    of their history. [AUTHOR] George Orwell [TAGS] propaganda history rewriting-history
  - I donât trust anybody. Not anybody. And the more that I care about someone, the
    more sure I am theyâre going to get tired of me and take off. [AUTHOR] Rainbow
    Rowell,  [TAGS] friendship fear tired trust
- source_sentence: If we find ourselves with a desire that nothing in this world can
    satisfy, the most probable explanation is that we were made for another world.
    [AUTHOR] C.S. Lewis [TAGS] god world
  sentences:
  - Eating and reading are two pleasures that combine admirably. [AUTHOR] C.S. Lewis
    [TAGS] eating-reading
  - Poetry is what happens when nothing else can. [AUTHOR] Charles Bukowski [TAGS]
    poetry
  - For those who believe in God, most of the big questions are answered. But for
    those of us who can't readily accept the God formula, the big answers don't remain
    stone-written. We adjust to new conditions and discoveries. We are pliable. Love
    need not be a command nor faith a dictum. I am my own god. We are here to unlearn
    the teachings of the church, state, and our educational system. We are here to
    drink beer. We are here to kill war. We are here to laugh at the odds and live
    our lives so well that Death will tremble to take us. [AUTHOR] Charles Bukowski
    [TAGS] religion atheism
- source_sentence: Do you hate people?ââœI don't hate them...I just feel better when
    they're not around. [AUTHOR] Charles Bukowski,  [TAGS] paraphrased misanthropy
    humour
  sentences:
  - Success is not how high you have climbed, but how you make a positive difference
    to the world. [AUTHOR] Roy T. Bennett,  [TAGS] inspiring life-lessons leader inspiration
    leadership inspire optimism inspirational-attitude life-quotes motivational positive
    inspirational-life living inspirational-quotes life optimistic life-and-living
    success inspirational positive-life positive-thinking positive-affirmation motivation
  - I loved you like a man loves a woman he never touches, only writes to, keeps little
    photographs of. [AUTHOR] Charles Bukowski,  [TAGS] love
  - The longer and more carefully we look at a funny story, the sadder it becomes.
    [AUTHOR] Nikolai V. Gogol [TAGS] humor sadness
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: quotes eval
      type: quotes-eval
    metrics:
    - type: pearson_cosine
      value: 0.9503529057381603
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.7867322921951185
      name: Spearman Cosine
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    "Do you hate people?ââœI don't hate them...I just feel better when they're not around. [AUTHOR] Charles Bukowski,  [TAGS] paraphrased misanthropy humour",
    'I loved you like a man loves a woman he never touches, only writes to, keeps little photographs of. [AUTHOR] Charles Bukowski,  [TAGS] love',
    'The longer and more carefully we look at a funny story, the sadder it becomes. [AUTHOR] Nikolai V. Gogol [TAGS] humor sadness',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity

* Dataset: `quotes-eval`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| pearson_cosine      | 0.9504     |
| **spearman_cosine** | **0.7867** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 3,731 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                          | label                                                          |
  |:--------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                              | string                                                                              | float                                                          |
  | details | <ul><li>min: 19 tokens</li><li>mean: 53.96 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 17 tokens</li><li>mean: 54.27 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.34</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                             | sentence_1                                                                                                                                               | label            |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>It's only after we've lost everything that we're free to do anything. [AUTHOR] Chuck Palahniuk,  [TAGS] loss</code>                                                                                                                              | <code>This is your life and its ending one moment at a time. [AUTHOR] Chuck Palahniuk,  [TAGS] life</code>                                               | <code>1.0</code> |
  | <code>I know not all that may be coming, but be it what it will, I'll go to it laughing. [AUTHOR] Herman Melville,  [TAGS] adventure laughter</code>                                                                                                   | <code>Love looks not with the eyes, but with the mind,And therefore is winged Cupid painted blind. [AUTHOR] William Shakespeare,  [TAGS] love</code>     | <code>0.0</code> |
  | <code>A childhood without books â that would be no childhood. That would be like being shut out from the enchanted place where you can go and find the rarest kind of joy. [AUTHOR] Astrid Lindgren [TAGS] children-s-literature children books</code> | <code>You cannot protect yourself from sadness without protecting yourself from happiness. [AUTHOR] Jonathan Safran Foer [TAGS] happiness sadness</code> | <code>0.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 2
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch | Step | quotes-eval_spearman_cosine |
|:-----:|:----:|:---------------------------:|
| 1.0   | 234  | 0.7807                      |
| 2.0   | 468  | 0.7867                      |


### Framework Versions
- Python: 3.10.0
- Sentence Transformers: 4.1.0
- Transformers: 4.52.3
- PyTorch: 2.7.0+cpu
- Accelerate: 1.7.0
- Datasets: 3.6.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->