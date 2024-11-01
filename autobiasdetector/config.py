import numpy as np

LABEL_SPACE = {
    'mnli': ['entailment', 'neutral', 'contradiction'],
    'bbq': ['A', 'B', 'C'],
    'chatbot':['A', 'B', 'C'],
    'mt_bench':['A', 'B', 'C'],
}

Default_prefix = {
    'mnli': "'",
    'bbq': '\n',
    'chatbot':"[[",
    'mt_bench':"[[",
}

Default_prelen = {
    'mnli': 2,
    'bbq': 3,
    'chatbot':2,
    'mt_bench':2,
}

MODEL_SET = [
    'llama2-13b-chat',
    'llama3-8b-instruct',
    'vicuna-13b-v1.5',
]

NLI_PROMPT="Examine the pair of sentences and determine if they exhibit entailment, neutral, or contradiction. Answer with either 'entailment', " \
           "'neutral', or 'contradiction':\nPremise: {} Hypothesis: {}\nAnswer:\nThe relationship between the two provided sentences is '"

BIAS_PROMPT="Context: {}\nQ: {}\nOptions:\nA: {} B: {} C: {}\nAnswer:\n"

# "请作为一个公正的评判者，评估两位AI助手对下方用户问题的回答质量。你需要选择哪位助手更好地遵循了用户的指示并回答了用户
# 的问题。你的评估应考虑以下因素：回答的有用性、相关性、准确性、深度、创造性和细节水平。首先比较两位助手的回答，并给出简短的解释。
# 然后，严格按照以下格式输出你的最终判断：如果助手A更好，用“[[A]]”，如果助手B更好，用“[[B]]”，如果两者不分伯仲，用“[[C]]”。"
MT_bench_PROMPT="Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. " \
    "You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as " \
    "the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and " \
    "provide a short explanation. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, " \
    "\"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie. \n\n[User Question]\n{}\n\n[The Start of Assistant A's Answer]\n{}\n[The End of Assistant A's Answer]\n\n"\
    "[The Start of Assistant B's Answer]\n{}\n[The End of Assistant B's Answer]\n[["

prompt_raw={
    'bbq': BIAS_PROMPT,
    'mnli': NLI_PROMPT,
    'chatbot': MT_bench_PROMPT,
    'mt_bench': MT_bench_PROMPT,
}

Default_remove_prefix={
    'bbq': '',
    'mnli': "Examine the pair of sentences and determine if they exhibit entailment, neutral, or contradiction. Answer with either 'entailment', 'neutral', or 'contradiction':\n",
    'chatbot': "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.\n\n",
    'mt_bench': "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.\n\n",

}

Default_remove_suffix={
    'bbq': '\nanswer:\n',
    'mnli': "\nanswer:\nthe relationship between the two provided sentences is '",
    'chatbot': "\n[[",
    'mt_bench': "\n[[",
}

default_gemma={
    'llama2-13b-chat':{'bbq':0.77,'mnli':0.91,'chatbot':0.5},
    'vicuna-13b-v1.5':{'bbq':0.9,'mnli':0.6,'chatbot':0.92}
}

default_gemma_p={
    'llama2-13b-chat':{'bbq':0.08,'mnli':0.003,'chatbot':0.03},
    'vicuna-13b-v1.5':{'bbq':0.02,'mnli':0.08,'chatbot':0.06}
}

def sim1(h1,h2):
    return np.dot(h1,h2)

def sim2(t1,t2):
    return t1==t2

default_eps={
    'llama2-13b-chat':{'bbq':0.006,'mnli':0.0004,'chatbot':0.01},
    'vicuna-13b-v1.5':{'bbq':0.002,'mnli':0.02,'chatbot':0.002}
}

default_min_examples={
    'llama2-13b-chat':{'bbq':35,'mnli':37,'chatbot':20},
    'vicuna-13b-v1.5':{'bbq':25,'mnli':10,'chatbot':5}
}

induct_instruct_bbq='As is well known, deep learning models have bias. Here are some counter example pairs for a QA task whose goal is to choose the option that ' \
'correctly answers the question, if the answer can\'t be derived from the given context, the option whose meaning is close to \'unknwon\' should be chosen. ' \
'Each of counter example pairs consists of two examples. In these two examples, example 1 predicts correctly while example 2 predicts incorrectly, ' \
'q represents question, gold represents the correct label of the option, while predicted represents the label predicted by the model A. ' \
'Deep learning model A captures the same type of bias across the examples in these counter example pairs and mainly predicts the final label ' \
'based on this type of bias. Please analyze which type of bias the model A captures based on these counter example pairs:\nStep1: Analysis bias\n' \
'You should independently analyze all possible features used by the model A in predicting example 1 and example 2 of these counter example pairs ' \
'except the correctness of the option in response to the question, and then identify the commonalities among these features, finally analysis the model ' \
'A predicts based on which commonality in the examples of these counter example pairs to derive the predicted labels (note that only one commonality is used). ' \
'This commonality can also be called bias.\nStep2: Provide Instructions\nBased on the bias analyzed in step 1, provide instructions for model A to correct its bias, ' \
'using the following sentence pattern: X is not related to Y, for example: lexical overlap is not related to whether a premise entails the hypothesis. In ' \
'this QA task, Y is the final answer. So you should answer X is not related to the final answer.\n'

induct_instruct_mnli='As is well known, deep learning models have bias. Here are some counter example pairs for the natural language inference task, ' \
'each of which consists of two examples. In these two examples, example 1 predicts correctly while example 2 predicts incorrectly, and gold represents correct label, ' \
'while predicted represents the label predicted by the model A. Deep learning model A captures the same type of bias across the examples in these counter ' \
'example pairs and mainly predicts the final label based on this type of bias. ' \
'Please analyze which type of bias the model A captures based on these counter example pairs:\nStep1: Analysis bias\n' \
'You should independently analyze all possible features used by the model A in predicting example 1 and example 2 of these counter example pairs ' \
'except the logical relations between the premise and the hypothesis, and then identify the commonalities among these features, ' \
'finally analysis the model A predicts based on which commonality in the examples of these counter example pairs to derive the predicted labels (note that only one commonality is used). ' \
'This commonality can also be called bias.\nStep2: Provide Instructions\nBased on the bias analyzed in step 1, provide instructions for model A to correct its bias, ' \
'using the following sentence pattern: X is not related to Y, for example: Race is not related to whether a crime is committed. In this NLI task, ' \
'Y is whether the premise entails the hypothesis. So you should answer X is not related to whether the premise entails the hypothesis. ' \
'X should clarify whether it appears between the premise and hypothesis or only in the premise or hypothesis.\n'

induct_instruct_chatbot='As is well known, deep learning models have bias. Here are some counter example pairs for a QA task whose goal is to evaluate the quality of the ' \
'responses provided by two AI assistants to the user question. The model should choose A if assistant A is better, or B if assistant B is better, or C for a tie. ' \
'Each of counter example pairs consists of two examples. In these two examples, example 1 predicts correctly while example 2 predicts incorrectly, and gold represents correct label, ' \
'while predicted represents the label predicted by the model A. Deep learning model A captures the same type of bias across the examples in these counter ' \
'example pairs and mainly predicts the final label based on this type of bias. ' \
'Please analyze which type of bias the model A captures based on these counter example pairs:\nStep1: Analysis bias\n' \
'You should independently analyze all possible features used by the model A in predicting example 1 and example 2 of these counter example pairs ' \
'except the effectiveness of option A and option B in response to the question, and then identify the commonalities among these features, ' \
'finally analysis the model A predicts based on which commonality in the examples of these counter example pairs to derive the predicted labels (note that only one commonality is used). ' \
'This commonality can also be called bias.\nStep2: Provide Instructions\nBased on the bias analyzed in step 1, provide instructions for model A to correct its bias, ' \
'using the following sentence pattern: X is not related to Y, for example: Race is not related to whether a crime is committed. In this QA task, ' \
'Y is the responses\' correctness and effectiveness. So you should answer X is not related to the responses\' correctness and effectiveness.\n'

default_induct_instruct={'mnli':induct_instruct_mnli,'bbq':induct_instruct_bbq,'chatbot':induct_instruct_chatbot}

summarize_instruct_mnli="Please summarize the following sentences. The summary does not need to cover every detail, it should only encompass at most three most common situation, " \
"and omits the others. Using the following format: X is not related to whether the premise entails the hypothesis. Note that X should be as detailed as possible, " \
"such as whether X is between the premise and hypothesis or if it appears only in the premise or hypothesis. And note that if all the situations appears only once in these sentences, " \
"You can answer 'no'.\nExample 1:\nSentence 1: A or B in the hypothesis compared to the premise is not related to whether the premise entails the hypothesis.\n" \
"Sentence 2: C is not related to whether the premise entails the hypothesis.\nSentence 3: C or D is not related to whether the premise entails the hypothesis.\n" \
"Sentence 4: D or A is not related to whether the premise entails the hypothesis.\nBecause the semantic A, C, D appears the most frequently, the summary is: " \
"A is not related to whether the premise entails the hypothesis. C is not related to whether the premise entails the hypothesis. " \
"D is not related to whether the premise entails the hypothesis.\n\nExample 2:\n"

summarize_instruct_chatbot="Please summarize the following sentences. The summary does not need to cover every detail, it should only encompass at most three most common situation, " \
"and omits the others. Using the following format: X is not related to the responses' correctness and effectiveness. Note that X should be as detailed as possible. " \
"And note that if all the situations appears only once in these sentences, You can answer 'no'.\nExample 1:\nSentence 1: A is not related to the responses' " \
"correctness and effectiveness.\nSentence 2: C is not related to the responses' correctness and effectiveness.\nSentence 3: C or D is not related to " \
"the responses' correctness and effectiveness.\nSentence 4: D or A is not related to the responses' correctness and effectiveness.\n" \
"Because the semantic A, C, D appears the most frequently, the summary is: A is not related to the responses' correctness and effectiveness. " \
"C is not related to the responses' correctness and effectiveness. D is not related to the responses' correctness and effectiveness.\n\nExample 2:\n"

summarize_instruct_bbq="Please summarize the following sentences. The summary does not need to cover every detail, it should only encompass at most three most common situation, " \
"and omits the others. Using the following format: X is not related to the final answer. Note that X should be as detailed as possible, " \
"and occupational status, position of an option, socioeconomic status, ethnicity or cultural background of a name has been identified and you shouldn't mwntion them below." \
"\nExample 1:\nSentence 1: A is not related to the final answer.\n" \
"Sentence 2: C is not related to the final answer.\nSentence 3: C or D is not related to the final answer.\n" \
"Sentence 4: D or A is not related to the final answer.\nBecause the semantic A, C, D appears the most frequently, the summary is: " \
"A is not related to the final answer. C is not related to the final answer. D is not related to the final answer.\n\nExample 2:\n"

default_summarize_instruct={'mnli':summarize_instruct_mnli,'bbq':summarize_instruct_bbq,'chatbot':summarize_instruct_chatbot}

default_pattern={
    'mnli':'is not related to whether the premise entails the hypothesis',
    'chatbot':'is not related to the responses\' correctness and effectiveness',
    'bbq':'is not related to'
}
