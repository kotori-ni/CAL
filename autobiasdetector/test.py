MT_bench_PROMPT="Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. " \
    "You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as " \
    "the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and " \
    "provide a short explanation. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, " \
    "\"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie. \n\n[User Question]\n{}\n\n[The Start of Assistant A's Answer]\n{}\n[The End of Assistant A's Answer]\n\n"\
    "[The Start of Assistant B's Answer]\n{}\n[The End of Assistant B's Answer]\n[["

print(MT_bench_PROMPT.format("AQ","BQ","CQ"))