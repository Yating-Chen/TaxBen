<!-- 标题部分 -->

<div style="width: 100%; height: 100px; text-align: center; background-color: #f4f4f4; padding: 20px 0;">
   <h1 style="font-size: 50px; font-weight: bold; color: black; line-height: 100px;">
       TaxBen: Benchmarking the Chinese Tax Knowledge of Large Language Models
   </h1>
</div>

   Welcome to here, let's get to know TaxBen together. </br>
   We introduce TaxBen, the first evaluation benchmark specifically designed for Chinese taxation. It carefully compiles a rare tax data and accompanying prompts, including 9 datasets totaling 6K samples , which are derived from manually collected data and model-assisted annotations. We divided these datasets into 3 skill levels according to widely accepted Bloom’s cognitive models: (1) Knowledge Memorization'(KM), (2) Knowledge Understanding(KU), and (3) Knowledge  Application(KA). Evaluating 18 LLMs reveals significant performance differences.

**Evaluations**:

> Knowledge Memorization (KM):

- [Tax Law Recitation (TaxRecite)](https://huggingface.co/datasets/TaxBen/TaxBen/blob/main/TaxRecite.json)

> Knowledge Understanding (KU):

- [Tax Topic Classification (TaxTopic)](https://huggingface.co/datasets/TaxBen/TaxBen/blob/main/TaxTopic.json)
- [Tax News Summarization (TaxSum)](https://huggingface.co/datasets/TaxBen/TaxBen/blob/main/TaxSum.json)
- [Tax Reading Comprehension (TaxRead)](https://huggingface.co/datasets/TaxBen/TaxBen/blob/main/TaxRead.json)

> Knowledge Application (KA):

- [Tax Payment Calculation (TaxCalc)](https://huggingface.co/datasets/TaxBen/TaxBen/blob/main/TaxCalc.json)
- [Tax Single-Choice Exam (TaxSCQ)](https://huggingface.co/datasets/TaxBen/TaxBen/blob/main/TaxSCQ.json)
- [Tax Multiple-Choice Exam (TaxMCQ)](https://huggingface.co/datasets/TaxBen/TaxBen/blob/main/TaxMCQ.json)
- [Tax Knowledge Q&A (TaxQA)](https://huggingface.co/datasets/TaxBen/TaxBen/blob/main/TaxQA.json)
- [Tax Board Q&A (TaxBoard)](https://huggingface.co/datasets/TaxBen/TaxBen/blob/main/TaxBoard.json)



### Innovation Points


- **Introducing TaxBen**, the first benchmark specifically designed for Chinese taxation, comprising 6K instances.
- **Developing a rare tax dataset**, created by domain experts and augmented with ChatGPT-assisted annotations, featuring highquality data and carefully designed prompts.
- **Establishing the taxonomy of tax tasks**, organizing the dataset according to Bloom’s cognitive taxonomy to assess capabilities in memorization, comprehension, and application.
- **Addressing the challenge of matching numerical predictions with labels**, exploring the difficulty of consistent manual prompts and utilizing ChatGPT to guide numerical extraction for effective evaluation.
- **Conducting in-depth evaluations of 18 popular LLMs**, revealing and discussing their strengths and limitations in tax-related tasks.
- **Multi-evaluation across tax and NLP tasks** helps analyze the LLM’s shortcomings and potential strengths, further enhancing its performance.

---

## TaxBen Evalution Benchmark result: The evaluation results of 18 representative large models on TaxBen.

### Tasks

|           **Tax Task**           | **Dataset** |     **Specific Task**     | **Scale** |       **Metric**       |       **Type**       |                **Method**                |
| :------------------------------: | :---------: | :-----------------------: | :-------: | :--------------------: | :------------------: | :--------------------------------------: |
| Tax Knowledge Memorization (KM)  |  TaxRecite  |    Tax Law Recitation     |    200    |  BERTScore, BARTScore  |   Generation (GEN)   | Human-chatGPT collaborative Construction |
| Tax Knowledge Understanding (KU) |   TaxSum    |  Tax News Summarization   |   1000    |  BERTScore, BARTScore  |   Generation (GEN)   | Human-chatGPT collaborative Construction |
|                                  |  TaxTopic   | Tax Topic Classification  |   1000    | Accuracy, F1, Macro F1 | Classification (CLS) |              Manual Created              |
|                                  |   TaxRead   | Tax Reading Comprehension |   1000    |      EM Accuracy       |   Generation (GEN)   | Human-chatGPT collaborative Construction |
| Tax Knowledge Understanding (KA) |   TaxCalc   |  Tax Payment Calculation  |    500    |      EM Accuracy       |   Reasoning (REA)    |              Manual Created              |
|                                  |   TaxSCQ    |  Tax Single-Choice Exam   |    700    | Accuracy, F1, Macro F1 | Classification (CLS) |              Manual Created              |
|                                  |   TaxMCQ    | Tax Multiple-Choice Exam  |    400    |      EM Accuracy       | Classification (CLS) |              Manual Created              |
|                                  |    TaxQA    |     Tax Knowledge Q&A     |    700    |  BERTScore, BARTScore  |   Generation (GEN)   | Human-chatGPT collaborative Construction |
|                                  |  TaxBoard   |       Tax Board Q&A       |    500    |  BERTScore, BARTScore  |   Generation (GEN)   | Human-chatGPT collaborative Construction |

### Zero-shot detailed results for 18 popular LLMs on TaxBen

| Task | Dataset   | Metrics   | ChatGPT   | Mistral-V0.3 | Gemma  | LLaMA3 | Bayling2 | Grok3      | DeepSeek-llm | Baichuan2 | Atom   | Qwen2.5   | ChineseLLaMA3 | ERNIE-3.5  | ChatCLM3 | Yi        | GLM4      | DeepSeek-R1 | InternLM2.5 | YaYi2  |
| ---- | --------- | --------- | --------- | ------------ | ------ | ------ | -------- | ---------- | ------------ | --------- | ------ | --------- | ------------- | ---------- | -------- | --------- | --------- | ----------- | ----------- | ------ |
| KM   | TaxRecite | BERTScore | 0.493     | 0.390        | 0.224  | 0.390  | 0.465    | 0.537      | 0.504        | 0.470     | 0.479  | 0.502     | 0.428         | **0.721**  | 0.493    | 0.489     | 0.481     | 0.454       | 0.518       | 0.491  |
|      |           | BARTScore | -5.166    | -5.903       | -7.357 | -5.657 | -5.331   | -4.991     | -5.162       | -5.306    | -5.304 | -5.044    | -5.367        | **-3.869** | -5.189   | -5.245    | -5.221    | -5.327      | -5.073      | -5.220 |
| KU   | TaxSum    | BERTScore | **0.624** | 0.043        | 0.234  | 0.360  | 0.524    | 0.620      | 0.541        | 0.413     | 0.464  | 0.471     | 0.552         | 0.618      | 0.336    | 0.549     | 0.331     | 0.536       | 0.432       | 0.331  |
|      |           | BARTScore | -4.378    | -6.826       | -7.288 | -5.934 | -5.003   | **-4.364** | -4.907       | -5.523    | -5.380 | -5.234    | -4.794        | -4.389     | -5.965   | -4.775    | -5.994    | -4.932      | -5.444      | -6.058 |
|      | TaxTopic  | Accuracy  | 0.442     | 0.152        | 0.000  | 0.222  | 0.054    | 0.346      | 0.128        | 0.146     | 0.211  | 0.340     | 0.193         | 0.424      | 0.116    | 0.210     | **0.630** | 0.226       | 0.434       | 0.004  |
|      |           | F1        | 0.402     | 0.092        | 0.000  | 0.094  | 0.084    | 0.368      | 0.071        | 0.083     | 0.093  | 0.356     | 0.095         | 0.416      | 0.147    | 0.097     | **0.618** | 0.143       | 0.437       | 0.007  |
|      |           | Macro F1  | 0.267     | 0.039        | 0.000  | 0.039  | 0.035    | 0.207      | 0.030        | 0.034     | 0.038  | 0.177     | 0.039         | 0.255      | 0.054    | 0.041     | **0.411** | 0.057       | 0.254       | 0.005  |
|      | TaxRead   | Accuracy  | 0.843     | 0.558        | 0.001  | 0.207  | 0.658    | 0.838      | 0.421        | 0.210     | 0.533  | **0.848** | 0.291         | 0.841      | 0.806    | 0.694     | 0.626     | 0.702       | 0.000       | 0.554  |
| KA   | TaxCalc   | Accuracy  | 0.046     | 0.000        | 0.000  | 0.002  | 0.004    | **0.190**  | 0.004        | 0.000     | 0.006  | 0.036     | 0.002         | 0.076      | 0.018    | 0.024     | 0.058     | 0.024       | 0.042       | 0.002  |
|      | TaxSCQ    | Accuracy  | 0.436     | 0.079        | 0.129  | 0.231  | 0.147    | 0.541      | 0.259        | 0.323     | 0.264  | 0.407     | 0.226         | 0.589      | 0.283    | 0.307     | 0.460     | 0.230       | **0.607**   | 0.346  |
|      |           | F1        | 0.436     | 0.067        | 0.166  | 0.128  | 0.145    | 0.538      | 0.209        | 0.319     | 0.199  | 0.394     | 0.092         | 0.589      | 0.279    | 0.251     | 0.475     | 0.133       | **0.614**   | 0.345  |
|      |           | Macro F1  | 0.437     | 0.072        | 0.167  | 0.134  | 0.148    | 0.538      | 0.213        | 0.320     | 0.202  | 0.395     | 0.098         | 0.589      | 0.281    | 0.255     | 0.474     | 0.138       | **0.614**   | 0.344  |
|      | TaxMCQ    | Accuracy  | 0.163     | 0.003        | 0.000  | 0.000  | 0.013    | 0.225      | 0.020        | 0.060     | 0.053  | 0.313     | 0.015         | 0.255      | 0.030    | **0.375** | 0.198     | 0.020       | 0.310       | 0.005  |
|      | TaxQA     | BERTScore | 0.477     | 0.373        | 0.243  | 0.407  | 0.477    | 0.487      | 0.480        | 0.458     | 0.460  | 0.508     | 0.457         | 0.531      | 0.491    | 0.479     | **0.538** | 0.434       | 0.513       | 0.508  |
|      |           | BARTScore | -4.922    | -5.651       | -7.287 | -5.446 | -5.091   | -4.675     | -4.860       | -4.979    | -5.051 | -4.555    | -5.098        | **-4.126** | -4.926   | -4.893    | -4.649    | -5.243      | -4.721      | -4.999 |
|      | TaxBoard  | BERTScore | 0.636     | 0.210        | 0.234  | 0.506  | 0.572    | 0.653      | 0.583        | 0.551     | 0.576  | 0.604     | 0.540         | **0.688**  | 0.611    | 0.578     | 0.598     | 0.574       | 0.584       | 0.532  |
|      |           | BARTScore | -4.771    | -6.452       | -7.289 | -5.330 | -4.991   | -4.541     | -4.882       | -5.007    | -4.918 | -4.766    | -5.135        | **-4.200** | -4.862   | -4.888    | -4.788    | -4.981      | -4.854      | -5.212 |

### One-shot detailed results for 18 popular LLMs on TaxBen

| Task | Dataset   | Metrics   | ChatGPT   | Mistral-V0.3 | Gemma  | LLaMA3 | Bayling2 | Grok3      | DeepSeek-llm | Baichuan2 | Atom   | Qwen2.5   | ChineseLLaMA3 | ERNIE-3.5  | ChatCLM3 | Yi     | GLM4   | DeepSeek-R1 | InternLM2.5 | YaYi2  |
| ---- | --------- | --------- | --------- | ------------ | ------ | ------ | -------- | ---------- | ------------ | --------- | ------ | --------- | ------------- | ---------- | -------- | ------ | ------ | ----------- | ----------- | ------ |
| KM   | TaxRecite | BERTScore | 0.510     | 0.398        | 0.218  | 0.374  | 0.482    | 0.547      | 0.470        | 0.484     | 0.467  | 0.495     | 0.401         | **0.666**  | 0.342    | 0.474  | 0.152  | 0.454       | 0.074       | 0.318  |
|      |           | BARTScore | -5.161    | -5.716       | -7.399 | -5.593 | -5.308   | -4.940     | -5.208       | -5.225    | -5.325 | -5.075    | -5.495        | **-4.281** | -5.986   | -5.214 | -7.014 | -5.326      | -7.448      | -6.191 |
| KU   | TaxSum    | BERTScore | **0.624** | 0.339        | 0.549  | 0.426  | 0.471    | 0.623      | 0.576        | 0.562     | 0.573  | 0.264     | 0.587         | 0.617      | 0.200    | 0.589  | 0.254  | 0.534       | 0.206       | 0.066  |
|      |           | BARTScore | -4.385    | -5.579       | -4.800 | -5.519 | -5.308   | **-4.348** | -4.630       | -4.638    | -4.670 | -6.406    | -4.566        | -4.418     | -6.739   | -4.512 | -6.412 | -4.936      | -6.707      | -7.483 |
|      | TaxTopic  | Accuracy  | 0.472     | 0.168        | 0.004  | 0.124  | 0.008    | 0.394      | 0.208        | 0.149     | 0.185  | 0.280     | 0.111         | **0.551**  | 0.182    | 0.176  | 0.370  | 0.322       | 0.017       | 0.001  |
|      |           | F1        | 0.456     | 0.125        | 0.008  | 0.107  | 0.108    | 0.421      | 0.113        | 0.120     | 0.097  | 0.323     | 0.097         | **0.545**  | 0.237    | 0.100  | 0.464  | 0.323       | 0.029       | 0.002  |
|      |           | Macro F1  | 0.273     | 0.052        | 0.003  | 0.039  | 0.052    | 0.228      | 0.047        | 0.060     | 0.040  | 0.199     | 0.037         | **0.305**  | 0.085    | 0.044  | 0.283  | 0.117       | 0.012       | 0.002  |
|      | TaxRead   | Accuracy  | 0.840     | 0.781        | 0.639  | 0.018  | 0.525    | 0.831      | 0.785        | 0.815     | 0.656  | 0.780     | 0.732         | **0.844**  | 0.592    | 0.781  | 0.093  | 0.785       | 0.758       | 0.611  |
| KA   | TaxCalc   | Accuracy  | 0.002     | 0.000        | 0.000  | 0.000  | 0.000    | 0.002      | 0.002        | 0.000     | 0.000  | 0.002     | 0.002         | **0.006**  | 0.000    | 0.000  | 0.000  | 0.000       | 0.000       | 0.000  |
|      | TaxSCQ    | Accuracy  | 0.444     | 0.241        | 0.234  | 0.224  | 0.271    | 0.569      | 0.241        | 0.279     | 0.221  | 0.490     | 0.230         | **0.590**  | 0.201    | 0.303  | 0.333  | 0.246       | 0.301       | 0.206  |
|      |           | F1        | 0.441     | 0.141        | 0.129  | 0.087  | 0.279    | 0.566      | 0.111        | 0.238     | 0.093  | 0.492     | 0.088         | **0.591**  | 0.236    | 0.251  | 0.451  | 0.204       | 0.313       | 0.249  |
|      |           | Macro F1  | 0.442     | 0.146        | 0.134  | 0.094  | 0.277    | 0.566      | 0.116        | 0.240     | 0.099  | 0.492     | 0.095         | **0.590**  | 0.236    | 0.254  | 0.452  | 0.207       | 0.315       | 0.249  |
|      | TaxMCQ    | Accuracy  | 0.058     | 0.018        | 0.005  | 0.018  | 0.030    | 0.035      | 0.055        | 0.050     | 0.043  | **0.060** | 0.020         | 0.035      | 0.010    | 0.035  | 0.025  | 0.010       | 0.030       | 0.000  |
|      | TaxQA     | BERTScore | 0.500     | 0.428        | 0.232  | 0.393  | 0.513    | 0.513      | 0.467        | 0.469     | 0.441  | **0.542** | 0.426         | 0.526      | 0.508    | 0.465  | 0.492  | 0.450       | 0.538       | 0.454  |
|      |           | BARTScore | -4.778    | -5.412       | -7.484 | -5.501 | -5.059   | -4.498     | -4.934       | -4.992    | -5.271 | -4.400    | -5.283        | **-4.227** | -4.948   | -4.924 | -4.820 | -5.103      | -4.810      | -5.563 |
|      | TaxBoard  | BERTScore | 0.640     | 0.482        | 0.232  | 0.496  | 0.588    | 0.660      | 0.576        | 0.566     | 0.543  | 0.596     | 0.547         | **0.676**  | 0.606    | 0.578  | 0.470  | 0.583       | 0.517       | 0.498  |
|      |           | BARTScore | -4.682    | -5.393       | -7.343 | -5.334 | -4.902   | -4.460     | -4.892       | -4.966    | -5.099 | -4.779    | -5.061        | **-4.282** | -4.803   | -4.840 | -5.474 | -4.927      | -5.246      | -5.418 |

### Evaluation

#### Preparation

##### Locally install

```bash
cd TaxBen
pip install -r requirements.txt
cd src/tax-evaluation
pip install -e .[multilingual]
```

#### Automated Task Assessment

Before evaluation, please download [BART checkpoint](https://drive.google.com/u/0/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&export=download) to `src/metrics/BARTScore/bart_score.pth`.

 For automated evaluation, please follow these instructions:

1. Huggingface Transformer

   To evaluate a model hosted on the HuggingFace Hub, use this command:

```bash
python eval.py \
    --model "hf-causal-llama" \
    --model_args "use_accelerate=True,pretrained=PoLylm-13B,tokenizer=PoLylm-13B,use_fast=False" \
    --tasks "TaxBen_TaxRecite"
```

More details can be found in the [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) documentation.


2. Commercial APIs

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python eval.py \
    --model chatgpt \
    --tasks TaxBen_TaxRecite
```

3. Self-Hosted Evaluation

To run inference backend:

```bash
bash scripts/run_interface.sh
```

Please adjust run_interface.sh according to your environment requirements.

To evaluate:

```bash
python data/*/evaluate.py
```

### Create new tasks

Creating a new task for TaxBen involves creating a Huggingface dataset and implementing the task in a Python file. This guide walks you through each step of setting up a new task using the TaxBen framework.

#### Creating your dataset in Huggingface

Your dataset should be created in the following format:

```python
{
    "query": "...",
    "answer": "...",
    "text": "..."
}
```

In this format:

- `query`: Combination of your prompt and text
- `answer`: Your label