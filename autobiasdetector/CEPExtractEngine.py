from .config import sim1,sim2,default_gemma,default_gemma_p
import json
import numpy as np
from tqdm import trange

def read_jsonl(filename):
    with open(filename, encoding='utf8') as f:
        datas = f.readlines()
    datas_tmp=[]
    for data in datas:
        data = json.loads(data)
        datas_tmp.append(data)
    return datas_tmp

def getrightwrong(datas, gemma_p, text_simfunc, alpha, beta, print_ave):
    right,wrong=[],[]
    gemma_p_ave,text_sim_ave,negative_num=0,0,0
    for data in datas:
        sim_value=text_simfunc(data['gold'],data['pred'])
        text_sim_ave+=sim_value
        if sim_value>beta:
            right.append(data)
        elif sim_value<alpha:
            gemma_p_ave+=data['gold_score']
            negative_num+=1
            if data['gold_score']<gemma_p:
                wrong.append(data)
    if print_ave:
        print('average relative loglikelihood probability of negative example: {}'.format(gemma_p_ave/negative_num))
        print('average similarity between predicted answer and gold answer: {}'.format(text_sim_ave/len(datas)))
        print(len(wrong))
    return right,wrong

class CEPEngine:
    def __init__(self, dataset_name, model_name, h_simfunc=sim1, text_simfunc=sim2):
        self.dataset_name=dataset_name
        self.model_name=model_name
        self.dir='data/'+dataset_name+'/'+model_name+'/'
        self.datas = read_jsonl(self.dir+'examples.jsonl')
        self.h_simfunc = h_simfunc
        self.text_simfunc = text_simfunc

    def extract(self,alpha=0.5,beta=0.5,gemma=None,gemma_p=None,write_file=True,print_ave=False):
        if gemma is None:
            if self.model_name in default_gemma and self.dataset_name in default_gemma[self.model_name]:
                gemma = default_gemma[self.model_name][self.dataset_name]
            else:
                raise('no default gemma for {} {}'.format(self.model_name,self.dataset_name))

        if gemma_p is None:
            if self.model_name in default_gemma_p and self.dataset_name in default_gemma_p[self.model_name]:
                gemma_p = default_gemma_p[self.model_name][self.dataset_name]
            else:
                raise('no default gemma_p for {} {}'.format(self.model_name,self.dataset_name))

        assert beta>=alpha
        negas = set()
        right_datas, wrong_datas = getrightwrong(self.datas, gemma_p, self.text_simfunc, alpha, beta, print_ave)
        data_pairs = []
        gemma_ave, gemma_num=0,0
        for i in trange(len(right_datas)):
            right_data = right_datas[i]
            v1 = np.array(right_data['representations'])
            for j, wrong_data in enumerate(wrong_datas):
                if self.text_simfunc(right_data['pred'],wrong_data['pred'])>beta and self.text_simfunc(right_data['gold'],wrong_data['gold'])<alpha:
                    v2 = np.array(wrong_data['representations'])
                    sim = self.h_simfunc(v1, v2)
                    gemma_ave+=sim
                    gemma_num+=1
                    if sim > gemma:
                        data_pairs.append((i, j))
                        negas.add(j)

        if print_ave is True:
            print('average similarity of hidden states between positive and negative examples: {}'.format(gemma_ave/gemma_num))
        print('num of counter example pairs: {} num of negative examples: {}'.format(len(data_pairs), len(negas)))
        if write_file:
            with open(self.dir + 'filtered_examples_relative.jsonl', 'w', encoding='utf-8') as f:
                for pair in data_pairs:
                    right_data = right_datas[pair[0]]
                    wrong_data = wrong_datas[pair[1]]
                    json.dump(({'gold':right_data['gold'],'pred':right_data['pred'],'content':right_data['content'],"representations":right_data["representations"]},
                               {'gold':wrong_data['gold'],'pred':wrong_data['pred'],'content':wrong_data['content'],"representations":wrong_data["representations"]}),
                               f,ensure_ascii=False)
                    f.write('\n')