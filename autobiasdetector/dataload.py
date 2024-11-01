import json
import datasets
from tqdm import tqdm
from .config import NLI_PROMPT, BIAS_PROMPT, MT_bench_PROMPT, prompt_raw, LABEL_SPACE

def readjsonl(filename):
    with open(filename, encoding='utf8') as f:
        datas = f.readlines()
    datas_tmp=[]
    for data in datas:
        data = json.loads(data)
        datas_tmp.append(data)
    return datas_tmp

class Dataset(object):
    def __init__(self):
        self.data = None
        self.dataset_name = None
        self.id2label = None
        self.name = None

    def __len__(self):
        assert self.data is not None, "self.data is None. Please load data first."
        return len(self.data)

    def get_content_by_idx(self, idx, *args):
        raise NotImplementedError("get_content_by_idx() must be implemented in the subclass.")

class MNLI(Dataset):
    def __init__(self, data_path, label_space, id2label, prompt=NLI_PROMPT, split='dev'):
        if split=='dev':
            matched = datasets.load_from_disk(data_path)["validation_matched"]
            mismatched = datasets.load_from_disk(data_path)["validation_mismatched"]
        elif split=='test':
            matched = datasets.load_from_disk(data_path)["test_matched"]
            mismatched = datasets.load_from_disk(data_path)["test_mismatched"]
        else:
            raise('illegal parameter')
        self.data = datasets.concatenate_datasets([matched, mismatched])
        self.prompt = prompt
        self.label_space=label_space
        self.name='mnli'
        self.id2label=id2label

    def get_content_by_idx(self, idx, *args):
        content = self.prompt.format(self.data[idx]['premise'].strip(' ').strip('.') + '.', self.data[idx]['hypothesis'].strip(' ').strip('.')+'.')
        label = self.id2label[self.data[idx]['label']]
        return {"content": content, "label": label,'label_space':self.label_space}

class BBQ(Dataset):
    def __init__(self, data_path, label_space, id2label, prompt=BIAS_PROMPT):
        datas=readjsonl(data_path+'/datas.jsonl')
        self.data=[]
        self.name = 'bbq'
        self.id2label = id2label
        self.label_space = label_space
        for data in tqdm(datas):
            context=data['context']
            question=data['question']
            all_choices=[data['ans0']+'.',data['ans1']+'.',data['ans2']+'.']
            label=data['label']
            self.data.append({"content":prompt.format(context,question,all_choices[0],all_choices[1],all_choices[2]),"label":self.id2label[label],'label_space':label_space})

    def get_content_by_idx(self, idx, task=None):
        return self.data[idx]

class Chatbot(Dataset):
    def __init__(self, data_path, label_space, id2label, prompt=MT_bench_PROMPT):
        datas=readjsonl(data_path+'/data.jsonl')
        self.data=[]
        self.name = 'chatbot'
        self.id2label = id2label
        self.label_space = label_space
        d={'model_a':0, 'model_b':1, 'tie':2}
        
        for data in tqdm(datas):
            q = data['q']
            a = data['a']
            b = data['b']
            self.data.append({"content": prompt.format(q, a, b), "label": self.id2label[d[data['judge']]], 'label_space':label_space})

    def get_content_by_idx(self, idx, task=None):
        return self.data[idx]

def create_dataset(dataset_name, data_dir, label_space=None, prompt=None, split='dev'):
    # label_space: None, list
    if prompt is None:
        prompt=prompt_raw[dataset_name]
    if label_space is None:
        label_space=LABEL_SPACE[dataset_name]
    elif type(label_space)!=list:
        raise('lillegal label space')

    id2label=dict()
    if label_space != 'undetermined':
        for i, label in enumerate(label_space):
            id2label[i]=label

    # label_space = ['A', 'B', 'C']
    # id2label = {0: 'A',1: 'B',2: 'C'}
    if dataset_name=='mnli':
        return MNLI(data_dir, label_space, id2label, prompt, split=split)
    elif dataset_name == 'bbq':
        return BBQ(data_dir, label_space, id2label, prompt)
    elif dataset_name == 'chatbot' or dataset_name == 'mt_bench':
        return Chatbot(data_dir, label_space, id2label, prompt)
    else:
        raise NotImplementedError
