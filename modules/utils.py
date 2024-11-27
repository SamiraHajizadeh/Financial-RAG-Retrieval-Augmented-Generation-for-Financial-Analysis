'''
Implementing some useful functions
'''

import re


def decompose_rag_response(responses):

  short_answers = []
  long_answers = []
  for res in responses:
    helpful_answer_match = re.search(r"Helpful Answer:\s*(.*?)\n", res, re.DOTALL)
    longer_explanation_match = re.search(r"Answer:\s*(.*?)\nIn summary", res, re.DOTALL)

    helpful_answer = helpful_answer_match.group(1) if helpful_answer_match else ""
    longer_explanation = longer_explanation_match.group(1) if longer_explanation_match else ""

    short_answers.append(helpful_answer)
    long_answers.append(longer_explanation)

  return short_answers, long_answers