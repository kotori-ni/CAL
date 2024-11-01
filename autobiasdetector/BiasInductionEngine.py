from .config import default_induct_instruct, default_summarize_instruct, default_pattern
import json
import random
from openai import OpenAI
from tqdm import tqdm

def readjsonl(filename):
    with open(filename, encoding='utf8') as f:
        datas = f.readlines()
    datas_tmp=[]
    for data in datas:
        data = json.loads(data)
        datas_tmp.append(data)
    return datas_tmp

def getids(todir):
    ids,l=[],[]
    datas = readjsonl(todir)
    num=0
    for data in datas:
        if type(data)==int:
            num=0
            l.append(data)
        else:
            if num%2==0:
                ids.append(data)
            num += 1
    return ids,l

def getbiasinfo(s,template):
    if template in s:
        start = 0
        position = s.find(template, start)
        tmp = position
        for i in range(position, -1, -1):
            if s[i] == '\n':
                tmp = i
                break
        return s[tmp + 1:position].lstrip("\"").lstrip("- ") + template
    return None

def getbias(fromdir,template):
    datas = readjsonl(fromdir)
    num=0
    bias={}
    pre=-1
    for data in datas:
        if type(data)==int:
            bias[data]=[]
            pre=data
            num=0
            continue
        if num%2==1:
            if template in data:
                bias[pre].append(getbiasinfo(data,template))
        num+=1
    return bias

class BiasInductionEngine:
    def __init__(self, model_name, dataset_name, induct_model_name, api_key, group_size=5):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.path = 'data/' + dataset_name + '/' + model_name + '/'
        if induct_model_name not in ['gpt-4o-2024-05-13','gpt-4-1106-preview','gpt-3.5-turbo-0125']:
            self.client = OpenAI(api_key="EMPTY",base_url="http://localhost:8000/v1")
        else:
            self.client = OpenAI(api_key=api_key)
        self.induct_model_name = induct_model_name
        self.grout_size=group_size
        self.fromdir = self.path + 'bias_clustered_relative.jsonl'
        self.todir = self.path + 'instruct_relative_' + self.induct_model_name + '.json'

    def callopenai(self, message, pattern):
        message = [{"role": "user", "content": message}]
        chat_response = self.client.chat.completions.create(model=self.induct_model_name, temperature=0,messages=message)
        reply = json.loads(chat_response.json())["choices"][0]["message"]["content"]
        stop_num = 5
        while stop_num > 0 and pattern not in reply:
            chat_response=self.client.chat.completions.create(model=self.induct_model_name,temperature=0.5,messages=message)
            reply = json.loads(chat_response.json())["choices"][0]["message"]["content"]
            stop_num -= 1
        return reply

    def induct(self, induct_instruct=None, pattern=None, max_samples=500, seed=0):
        induct_instruct=default_induct_instruct[self.dataset_name] if induct_instruct is None else induct_instruct
        pattern=default_pattern[self.dataset_name] if pattern is None else pattern
        f = open(self.todir, 'a+', encoding='utf-8')
        ids,l = getids(self.todir)
        orig_datas = readjsonl(self.fromdir)
        datas_classes = orig_datas[0]
        random.seed(seed)
        for key in datas_classes.keys():
            if int(key)>=1:
                continue
            if int(key) not in l:
                json.dump(int(key), f, ensure_ascii=False)
                f.write('\n')
                if int(key) in l[:-1]:
                    continue
            datas = datas_classes[key]
            if len(datas) > max_samples:
                datas = random.sample(datas, max_samples)
            else:
                datas = datas[:len(datas) // self.grout_size * self.grout_size]
            prompt = ''
            num = 0
            for i, data in tqdm(enumerate(datas)):
                num += 1
                if num <= self.grout_size:
                    prompt += '\n<counter example pair ' + str(num) + '>\nExample 1: ' + data[0][0] + '\ngold: ' + data[0][1] + '. predicted: ' + data[0][2] + \
                                  '.\nExample 2: ' + data[1][0] + '\ngold: ' + data[1][1] + '. predicted: ' + data[1][2] + '.\n'
                if num == self.grout_size:
                    message = induct_instruct + prompt
                    if message in ids:
                        num = 0
                        prompt = ''
                        continue
                    reply=self.callopenai(message,pattern)
                    json.dump(message, f, ensure_ascii=False)
                    f.write('\n')
                    json.dump(reply, f, ensure_ascii=False)
                    f.write('\n')
                    f.flush()
                    prompt = ''
                    num = 0
        print('induct finished')

    def summarize(self,summarize_instruct=None, pattern=None):
        summarize_instruct = default_summarize_instruct[self.dataset_name] if summarize_instruct is None else summarize_instruct
        pattern = default_pattern[self.dataset_name] if pattern is None else pattern
        bias = getbias(self.todir, pattern)
        bias_pattterns=[]
        for key in bias.keys():
            print(key)
            num = 1
            prompt = ''
            for i, bia in enumerate(bias[key]):
                prompt += 'sentence ' + str(num) + ': ' + bia + '.\n'
                num += 1

            if self.induct_model_name=='Qwen1.5-72B-Chat':
                message = summarize_instruct + prompt+'The summary is:'
            else:
                message = summarize_instruct + prompt
            chat_response = self.client.chat.completions.create(model=self.induct_model_name,temperature=0,messages=[{"role": "user", "content": message}])
            reply = json.loads(chat_response.json())["choices"][0]["message"]["content"]
            print(reply)
            bias_pattterns.append(reply)
        return bias_pattterns
