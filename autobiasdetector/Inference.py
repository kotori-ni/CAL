import torch
from vllm import LLM, SamplingParams
from vllm.sequence import SequenceData, SequenceGroupMetadata
from tqdm import tqdm
import json
from .config import Default_prefix, Default_prelen, Default_remove_prefix, Default_remove_suffix, MODEL_SET
import os

def write_jsonl(file,train_datas):
    with open(file, 'w', encoding='utf-8') as f:
        for data in train_datas:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

class InferenceEngine:
    def __init__(self, model_name, dataset, model_path=None, dtype=torch.float16, tensor_parallel_size=1):
        self.model_name = model_name
        assert self.model_name in MODEL_SET
        self.dataset = dataset
        self.model = LLM(model=self.model_name if model_path is None else model_path, tensor_parallel_size=tensor_parallel_size, dtype=dtype, max_model_len=4096)
        self.tokenizer = self.model.llm_engine.tokenizer.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '[PAD]'
            if self.model_name=='llama3-8b-instruct':
                self.tokenizer.pad_token=self.tokenizer.eos_token

        self.tokenizer.padding_side = 'left'
        self.device = torch.device('cuda')
        self.sampling_params = SamplingParams(temperature=0, top_p=1, top_k=1)

        # 初始化Softmax函数，用于沿着最后一个维度计算概率分布
        self.softmax = torch.nn.Softmax(dim=-1)
        self.num_layers = self.model.llm_engine.model_config.get_num_layers(self.model.llm_engine.driver_worker.model_runner.parallel_config)

    def getpreferedoutput(self, num_examples=5, return_type='token'):
        ids_list=[]
        for idx in tqdm(range(num_examples)):
            raw_data = self.dataset.get_content_by_idx(idx)
            input_text = raw_data['content']
            ids=self.model.generate(input_text,self.sampling_params)[0].outputs[0].token_ids
            ids_list.append(ids)
        if return_type=='token':
            return [self.tokenizer.convert_ids_to_tokens(ids_list[i]) for i in range(len(ids_list))]
        elif return_type=='ids':
            return ids_list
        else:
            raise('error')

    def getlabelrep(self, prefix, return_type='token'):
        label_spaces=self.dataset.label_space
        self.tokenizer.padding_side = 'right'
        length = self.tokenizer([prefix + label_space for label_space in label_spaces], padding=False, return_length=True)["length"]
        length = max(length)
        label_spaces_ids = self.tokenizer([prefix + label_space for label_space in label_spaces], padding="max_length",
                                          max_length=length, return_tensors="pt")["input_ids"]
        self.tokenizer.padding_side = 'left'
        if return_type=='token':
            return [self.tokenizer.convert_ids_to_tokens(label_spaces_ids[i]) for i in range(len(label_spaces_ids))]
        elif return_type=='ids':
            return label_spaces_ids
        else:
            raise('illegal return type')

    def process_input(self, raw_data, prefix, prelen):
        input_text = raw_data["content"]
        label = raw_data["label"]
        label_spaces = self.dataset.label_space

        # 为每个标签空间项附加前缀prefix后，计算这些文本的长度，选择这些长度的最大值，以用于后续的固定长度填充
        self.tokenizer.padding_side = 'right'
        length = self.tokenizer([prefix + label_space for label_space in label_spaces], padding=False, return_length=True)["length"]
        length = max(length)

        # 使用tokenizer对每个标签（加上前缀）进行标记化，使用最大长度length并用“max_length”模式填充，以确保所有标记序列具有相同的长度，返回的结果是词汇表的索引形式(input_ids)
        label_spaces_ids = self.tokenizer([prefix + label_space for label_space in label_spaces], padding="max_length", max_length=length,
                       return_tensors="pt")["input_ids"]
        
        self.tokenizer.padding_side = 'left'
        label_spaces_ids = torch.tensor(label_spaces_ids, dtype=torch.int32).to(self.device)

        # 剔除前缀部分标记，以便后续处理
        label_spaces_ids = label_spaces_ids[:, prelen:]

        return input_text, label, label_spaces_ids

    def inference(self, prefix=None, prelen=None, remove_prefix=None, remove_suffix=None, write_file=True):
        # prefix: 'chatbot':"[["
        # prelen: 'chatbot':2
        if prefix is None:
            prefix = Default_prefix[self.dataset.name]
            prelen = Default_prelen[self.dataset.name]
        elif prelen is None:
            prelen = self.tokenizer(prefix, padding=False, return_length=True)["length"]

        """
        # remove_prefix和remove_suffix用于从输入文本中去除指定的前后缀

        # remove_prefix
        'chatbot': "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to \
                    the user question displayed below. You should choose the assistant that follows the user's instructions and \
                    answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, \
                    accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two \
                    responses and provide a short explanation. After providing your explanation, output your final verdict by strictly \
                    following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a \
                    tie.\n\n",
        'chatbot': "请作为公正的评审，评估两个人工智能助手对下面用户问题的回答质量。您应该选择一个更好地遵循用户指示并回答用户问题的助手。 \
                    您的评估应考虑有用性、相关性、准确性、深度、创造力和响应的细节水平等因素。开始评估时，请比较两个回答，并提供简要解释。 \
                    在提供解释后，请严格遵循以下格式给出您的最终判决：如果助手 A 更好，请输出 "[[A]]"；如果助手 B 更好，请输出 "[[B]]"； \
                    如果平局，请输出 "[[C]]"。\n\n",

        # remove_suffix
        'chatbot': "\n[[",
        """

        remove_prefix = Default_remove_prefix[self.dataset.name] if remove_prefix is None else remove_prefix
        remove_suffix = Default_remove_suffix[self.dataset.name] if remove_suffix is None else remove_suffix

        data_len = len(self.dataset)

        # preds: 预测结果
        # gts: 真实结果
        # examples: 示例列表
        preds, gts, examples = [], [], []

        for idx in tqdm(range(data_len)):
            # These two pieces of data are too long, so remove these two pieces of data
            if self.dataset.name == 'chatbot' and (idx == 2635 or idx == 8995):
                continue
            example = {}
            raw_data = self.dataset.get_content_by_idx(idx)

            input_text, gt, label_spaces_ids = self.process_input(raw_data, prefix, prelen)
            
            example['content'] = input_text[len(remove_prefix):0-len(remove_suffix)].lower()
            if idx == 0:
                print(input_text)
                print()
            example['gold'] = gt
            gts.append(gt)

            prompt_token_ids = self.model.llm_engine.tokenizer.encode(input_text)
            seq_data = SequenceData(prompt_token_ids)
            group_id = 1
            seq = SequenceGroupMetadata(request_id=str(group_id),is_prompt=True,seq_data={group_id: seq_data},
                sampling_params=self.sampling_params,block_tables=None)
            seqs = [seq]
            input_ids, input_positions, input_metadata, _, _, _ = self.model.llm_engine.driver_worker.model_runner.prepare_input_tensors(seqs)
            option_num, max_seq_len = label_spaces_ids.shape
            input_ids_tmp = torch.zeros((option_num, input_ids.shape[1]), dtype=torch.long).to(self.device)
            for j in range(option_num):
                input_ids_tmp[j] = input_ids
            total_scores = torch.zeros((option_num, max_seq_len), dtype=torch.float).to(self.device)
            input_ids = input_ids_tmp

            for j in range(max_seq_len):
                with torch.no_grad():
                    hidden_states = self.model.llm_engine.driver_worker.model_runner.model(input_ids=input_ids[:1] if j == 0 else input_ids,
                        input_metadata=input_metadata, positions=input_positions, kv_caches=[(None, None)] * self.num_layers)

                    logits = self.model.llm_engine.driver_worker.model_runner.model.sampler._get_logits(hidden_states,
                                    self.model.llm_engine.driver_worker.model_runner.model.lm_head.weight,None)
                    logits = self.softmax(logits)

                if j == 0:
                    representations = self.softmax(hidden_states[0][-1]).cpu().tolist()
                    example['representations'] = representations
                    for k in range(option_num):
                        total_scores[k][j] = logits[0][-1][label_spaces_ids[k][j]]
                else:
                    for k in range(option_num):
                        if label_spaces_ids[k][j] != self.model.llm_engine.tokenizer.tokenizer.pad_token_id:
                            total_scores[k][j] = logits[k][-1][label_spaces_ids[k][j]]
                        else:
                            total_scores[k][j] = 1
                input_ids = torch.cat([input_ids, label_spaces_ids[:, j].view(-1, 1)], dim=-1).long().to('cpu').tolist()
                seqs = []
                for k in range(len(input_ids)):
                    seq_data = SequenceData(input_ids[k])
                    group_id = 1
                    seq = SequenceGroupMetadata(
                        request_id=str(group_id),
                        is_prompt=True,
                        seq_data={group_id: seq_data},
                        sampling_params=self.sampling_params,
                        block_tables=None,
                    )
                    seqs.append(seq)
                input_ids, input_positions, input_metadata, _, _, _ = self.model.llm_engine.driver_worker.model_runner.prepare_input_tensors(seqs)

            total_scores = torch.sum(torch.log(total_scores), dim=-1)
            pred = torch.argmax(total_scores, dim=-1).item()

            # For Influential Criterion:
            # The greater the loglikelihood probability of a gold answer compared to a predicted answer, the greater the
            # impact of bias information on LLM. Because the loglikelihood probability is negative, the expression
            # for the degree to which log(P(gold)) is greater than log(P(pred)) is log(P(pred))/log(P(gold))
            example['gold_score']=total_scores[pred].item()/total_scores[self.dataset.label_space.index(gt)].item()
            example['pred'] = self.dataset.id2label[pred]
            examples.append(example)
            preds.append(self.dataset.id2label[pred])

        score = self.eval(preds, gts)
        if write_file is True:
            if not os.path.exists('data/' + self.dataset.name + '/'+self.model_name):
                os.makedirs('data/' + self.dataset.name + '/'+self.model_name)
            write_jsonl('data/' + self.dataset.name + '/'+self.model_name+'/examples.jsonl', examples)
        return score

    def eval(self, preds, gts):
        if self.dataset.name in ["mnli", 'bbq', 'chatbot','mt_bench']:
            if not isinstance(preds, list):
                preds = [preds]
                gts = [gts]
            return sum(a == b for a, b in zip(preds, gts)) / len(preds)
        else:
            raise NotImplementedError("Eval this dataset {self.args.dataset} is not implemented!")

