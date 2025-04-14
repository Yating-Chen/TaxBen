import os
import re
import time
import evaluate
import numpy as np
from openai import OpenAI
from lm_eval.base import Task, rf
from lm_eval.metrics import mean, bleu, chrf, ter
from .utils import process_text
from .zhutils import process_zhtext
from seqeval.metrics import f1_score as entity_score
from sklearn.metrics import f1_score, matthews_corrcoef, mean_squared_error
from bart_score import BARTScorer


class Classification(Task):
    CALCULATE_MCC = True
    LOWER_CASE = True
    VERSION = 1
    EVAL_LAST_TURN = True

    def reformulate_turn_req(self, req, turn_request, turn):
        return req

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def construct_requests(self, doc, ctx):

        cont_request = rf.greedy_until(ctx, {"until": None})
        return cont_request

    def doc_to_decontamination_query(self, doc):
        return doc["text"]

    def doc_to_text(self, doc):

        return doc["query"]

    def doc_to_target(self, doc):

        return doc["answer"]

    def process_results(self, doc, results):
        gold: str = doc["choices"][doc["gold"]]
        if self.LOWER_CASE:
            gold = gold.lower()
        ini_result = results[0].strip()
        if self.LOWER_CASE:
            ini_result = ini_result.lower()

        result = None
        for choice in doc["choices"]:
            if self.LOWER_CASE:
                choice = choice.lower()
            if choice in ini_result:
                result = choice
                break
        if result is None:
            result = "missing"

        acc = 1.0 if gold == result else 0.0

        results = {
            "acc": acc,
            "missing": int(result == "missing"),
            "f1": (result, gold),
            "macro_f1": (result, gold),
        }

        if self.CALCULATE_MCC:
            results["mcc"] = (result, gold)

        return results

    def higher_is_better(self):
        metrics = {
            "acc": True,
            "f1": True,
            "macro_f1": True,
            "missing": False,
        }
        if self.CALCULATE_MCC:
            metrics["mcc"] = True
        return metrics

    def weighted_f1(self, items):
        preds, golds = zip(*items)
        labels = list(set(golds))
        preds = np.array(preds)
        golds = np.array(golds)
        f1 = f1_score(golds, preds, average="weighted", labels=labels)
        return f1

    def macro_f1(self, items):
        preds, golds = zip(*items)
        labels = list(set(golds))
        preds = np.array(preds)
        golds = np.array(golds)
        f1 = f1_score(golds, preds, average="macro", labels=labels)
        return f1

    def matthews_corrcoef(self, items):
        preds, golds = zip(*items)
        labels = {label: i for i, label in enumerate(list(set(golds)))}
        preds = [labels.get(pred, -1) for pred in preds]
        golds = [labels.get(gold, -1) for gold in golds]
        return matthews_corrcoef(golds, preds)

    def aggregation(self):
        metrics = {
            "acc": mean,
            "missing": mean,
            "f1": self.weighted_f1,
            "macro_f1": self.macro_f1,
        }
        if self.CALCULATE_MCC:
            metrics["mcc"] = self.matthews_corrcoef
        return metrics


class TaxSCQ(Classification):
    DATASET_PATH = "DATASET_PATH"


class TaxTopic(Classification):
    DATASET_PATH = "DATASET_PATH"


class AbstractiveSummarization(Task):
    VERSION = 1
    DATASET_NAME = None
    EVAL_LAST_TURN = True

    def reformulate_turn_req(self, req, turn_request, turn):
        return req

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return doc["answer"]

    def process_results(self, doc, results):
        return {
            "rouge1": (doc["answer"], results[0]),
            "rouge2": (doc["answer"], results[0]),
            "rougeL": (doc["answer"], results[0]),
            "bert_score_f1": (doc["answer"], results[0]),
            "bart_score": (doc["answer"], results[0]),
        }

    def higher_is_better(self):
        return {
            "rouge1": True,
            "rouge2": True,
            "rougeL": True,
            "bert_score_f1": True,
            "bart_score": True,
        }

    def construct_requests(self, doc, ctx):
        cont_request = rf.greedy_until(ctx, {"until": None})
        return cont_request

    def rouge_score(self, items):
        golds, preds = zip(*items)
        rouge = evaluate.load("src/metrics/rouge")
        results = rouge.compute(predictions=preds, references=golds)
        return results

    def rouge1(self, items):
        results = self.rouge_score(items)
        return results["rouge1"]

    def rouge2(self, items):
        results = self.rouge_score(items)
        return results["rouge2"]

    def rougeL(self, items):
        results = self.rouge_score(items)
        return results["rougeL"]

    def bert_score(self, items):
        if getattr(self, "_cache_bertscore", None) is None:
            golds, preds = zip(*items)
            bertscore = evaluate.load("src/metrics/bertscore")

            self._cache_bertscore = bertscore.compute(
                predictions=preds,
                references=golds,
                num_layers=5,
                lang="zh",
                model_type="/data1/hugang/PublicLLMs/chinese-xlnet-base/",
            )

            # self._cache_bertscore = bertscore.compute(
            #     predictions=preds,
            #     references=golds,
            #     model_type="/gemini/pretrain2/bert-base-multilingual-cased",
            # )

            return self._cache_bertscore
        else:
            return self._cache_bertscore

    def bert_score_f1(self, items):
        res = self.bert_score(items)
        return sum(res["f1"]) / len(res["f1"])

    def bart_score(self, items):
        golds, preds = zip(*items)
        bart_scorer = BARTScorer(device="cuda", checkpoint="/data1/hugang/PublicLLMs/bart-large-cnn/")
        bart_scorer.load(path="/data1/hugang/PublicLLMs/bart_score.pth")
        res = bart_scorer.score(srcs=preds, tgts=golds, batch_size=8)
        return sum(res) / len(res)

    def aggregation(self):
        return {
            "rouge1": self.rouge1,
            "rouge2": self.rouge2,
            "rougeL": self.rougeL,
            "bert_score_f1": self.bert_score_f1,
            "bart_score": self.bart_score,
        }


class TaxBoard(AbstractiveSummarization):
    DATASET_PATH = "DATASET_PATH"


class TaxQA(AbstractiveSummarization):
    DATASET_PATH = "DATASET_PATH"


class TaxRecite(AbstractiveSummarization):
    DATASET_PATH = "DATASET_PATH"


class TaxSum(AbstractiveSummarization):
    DATASET_PATH = "DATASET_PATH"


class MatchAnswer(Task):
    VERSION = 1
    DATASET_NAME = None
    EVAL_LAST_TURN = True

    def reformulate_turn_req(self, req, turn_request, turn):
        return req

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["text"]

    def doc_to_text(self, doc):
        return doc["query"]

    def construct_requests(self, doc, ctx):
        cont_request = rf.greedy_until(ctx, {"until": None})
        return cont_request

    def doc_to_target(self, doc):
        return doc["answer"]

    def process_results(self, doc, results):
        gold = doc["answer"]

        match = re.findall(r'[ABCDE](?:\W*[ABCDE]){1,4}', results[0])

        pres = re.sub(r'[^A-Z]', '', match[0]) if match else 'null'

        acc = 1.0 if pres == gold else 0.0

        return {
            "acc": acc,
        }

    def higher_is_better(self):
        return {
            "acc": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
        }


class TaxMCQ(MatchAnswer):
    DATASET_PATH = "DATASET_PATH"


class ComputeAnswer(Task):
    VERSION = 1
    DATASET_NAME = None
    EVAL_LAST_TURN = True

    def reformulate_turn_req(self, req, turn_request, turn):
        return req

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["text"]

    def doc_to_text(self, doc):
        return doc["query"]

    def construct_requests(self, doc, ctx):
        cont_request = rf.greedy_until(ctx, {"until": None})
        return cont_request

    def doc_to_target(self, doc):
        return doc["answer"]

    def process_results(self, doc, results):
        # 确保 gold 保留两位小数
        try:
            gold = float(doc["answer"])  # 将 gold 转换为浮点数
            formatted_gold = "{:.2f}".format(gold)  # 格式化为保留两位小数的字符串
        except ValueError:
            print("标准答案 (gold) 格式不正确，无法转换为浮点数")
            return {"acc": 0.0}  # 如果 gold 格式错误，直接返回准确率为 0

        # 初始化客户端
        client = OpenAI(
            base_url="https://api.xty.app/v1",
            api_key='',
            timeout=120
        )

        try:
            # 尝试直接比较结果
            pres = round(float(results[0]), 2)
            formatted_pres = "{:.2f}".format(pres)  # 格式化为保留两位小数的字符串
            acc = 1.0 if formatted_pres == formatted_gold else 0.0
            return {"acc": acc}

        except Exception as e:
            print(f"直接比较失败: {e}, 尝试通过GPT提取答案...")

            query = re.search(r'问题为：(.+)', doc['query']).group(1)

            print('query\n', query)
            print('logit\n', results[0])

            input = f"你将会收到[描述文本]，由一个<问题>和一个包含该问题的<答案段落>构成，你需要从段落中提取问题对应的答案，如果段落中没有包含问题的答案，则输出“未知”即可，不要自己回答，一定不需要有任何多余的解释。以下是一个例子：<问题>：'本月应该缴纳的增值税额为xx万元？' <答案段落>：'...本月应该缴纳的增值税额为10.00万元。' 提取出的答案为：'10.00万元' [描述文本]:  <问题>：{query}  <答案段落>：{results[0]} "

            print('input\n', input)

            while True:  # 无限循环，直到没有错误
                try:
                    # 调用 GPT 模型提取答案
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        max_tokens=500,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": input}
                        ]
                    )

                    answer = response.choices[0].message.content.strip()

                    print('GPT返回的答案\n', answer)

                    # 提取特定模式前后的数字
                    extracted_number = None

                    # 查找 "万元" 或 "元" 前面的数字
                    match_wan_yuan = re.search(r'([-+]?\d*\.?\d+)\s?(?=万元|元)', answer)
                    # 查找 "百分之" 后面的数字
                    match_percent = re.search(r'(?<=百分之)\d*\.?\d+', answer)

                    if match_wan_yuan:
                        extracted_number = match_wan_yuan.group(1)
                    elif match_percent:
                        extracted_number = match_percent.group(0)
                    else:
                        print("未找到符合条件的数字")
                        formatted_pres = "未知"

                    if extracted_number is not None:
                        pres = float(extracted_number)  # 转换为浮点数
                        formatted_pres = "{:.2f}".format(pres)  # 格式化为保留两位小数的字符串

                    print('提取并处理后的数字\n', formatted_pres)

                    # 计算准确率
                    acc = 1.0 if formatted_pres == formatted_gold else 0.0

                    return {"acc": acc}

                except Exception as e:
                    print(f"发生错误: {e}, 正在重新尝试...")
                    time.sleep(2)  # 等待 2 秒后再尝试

    def higher_is_better(self):
        return {
            "acc": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
        }


class TaxCalc(ComputeAnswer):
    DATASET_PATH = "DATASET_PATH"


class ReadComprehens(Task):
    VERSION = 1
    DATASET_NAME = None
    EVAL_LAST_TURN = True

    def reformulate_turn_req(self, req, turn_request, turn):
        return req

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["text"]

    def doc_to_text(self, doc):
        return doc["query"]

    def construct_requests(self, doc, ctx):
        cont_request = rf.greedy_until(ctx, {"until": None})
        return cont_request

    def doc_to_target(self, doc):
        return doc["answer"]

    def process_results(self, doc, results):
        gold = doc["answer"]

        acc = 1.0 if gold in results[0] else 0.0

        return {
            "acc": acc,
        }

    def higher_is_better(self):
        return {
            "acc": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
        }


class TaxRead(ReadComprehens):
    DATASET_PATH = "DATASET_PATH"



