"""
    This file takes a CoQA data file as input and generates the input files for the conversational models.
"""

import argparse
import json
import time
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9020')

def _str(s):
    """ Convert PTB tokens to normal tokens """
    if (s.lower() == '-lrb-'):
        s = '('
    elif (s.lower() == '-rrb-'):
        s = ')'
    elif (s.lower() == '-lsb-'):
        s = '['
    elif (s.lower() == '-rsb-'):
        s = ']'
    elif (s.lower() == '-lcb-'):
        s = '{'
    elif (s.lower() == '-rcb-'):
        s = '}'
    return s

def tokenize_text(text):
    paragraph = nlp.annotate(text, properties={
                             'annotators': 'tokenize, ssplit',
                             'outputFormat': 'json'})
    tokens = []
    for sent in paragraph['sentences']:
        for token in sent['tokens']:
            tokens.append(_str(token['word']))
    return ' '.join(tokens)

def get_answer(qa):
    # 1. Take Model Answer (answers), if "answers" is present
    if 'answers' in qa.keys() and len(qa['answers']) > 0:
        key = 'answers'
    else:
        key = 'plausible_answers'

    if len(qa[key]) == 0:
        return False, ""
    
    # 2. Compute the frequency of answers
    answers_dict = {}
    for answer in qa[key]:
        answer_str = answer['text']
        if answer_str not in answers_dict:
            answers_dict[answer_str] = 0
        answers_dict[answer_str] += 1

    # Sort the answers in term of frequency
    answers_list = sorted(answers_dict.items(), key=lambda x: x[1], reverse=True)

    # Check if it is the ONLY answer available
    if len(answers_list) == 1: 
        pass
    # Check if the first answer has the majority votes
    elif answers_list[0][1] > answers_list[1][1]:  
        pass
    else:
        # Retrieve list of answers with same frequencies
        highest_frequency = answers_list[0][1]
        answers_list = [x for x in answers_list if x[1] == highest_frequency]

        # Sort the answer based on the length
        answers_list.sort(key=lambda x: len(x[0]), reverse=True)

    return True, answers_list[0][0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', '-d', type=str, required=True)
    parser.add_argument('--n_history', type=int, default=0,
                        help='leverage the previous n_history rounds of Q/A pairs'
                             'if n_history == -1, use all history')
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--output_file', '-o', type=str, required=True)
    args = parser.parse_args()

    f_src = open('{}-h{}-src.txt'.format(args.output_file, args.n_history), 'w')
    f_tgt = open('{}-h{}-tgt.txt'.format(args.output_file, args.n_history), 'w')

    with open(args.data_file) as f:
        dataset = json.load(f)

    start_time = time.time()
    data = []

    sanity_checks = {}
    impossible_questions = []
    for i, datum in enumerate(dataset['data']):
        if i % 10 == 0:
            print('processing %d / %d (used_time = %.2fs)...' %
                  (i, len(dataset['data']), time.time() - start_time))

        for paragraph in datum['paragraphs']:
            context_str = tokenize_text(paragraph['context'])
            
            history = []
            for qa in paragraph['qas']:

                # Tokenize Question (Same As Original)
                question = qa['question']
                question_str = tokenize_text(question)

                # Simple Heuristic To Determine Answers
                has_answer, answer_str = get_answer(qa)

                if not has_answer:
                    impossible_questions.append(qa['id'])

                full_str = context_str + ' ||'
                if args.n_history < 0:
                    for i, (q, a) in enumerate(history):
                        d = len(history) - i
                        full_str += ' <Q{}> '.format(d) + q + ' <A{}> '.format(d) + a
                elif args.n_history > 0:
                    context_len = min(args.n_history, len(history))
                    for i, (q, a) in enumerate(history[-context_len:]):
                        d = context_len - i
                        full_str += ' <Q{}> '.format(d) + q + ' <A{}> '.format(d) + a
                full_str += ' <Q> ' + question_str
                
                if args.lower:
                    full_str = full_str.lower()
                    answer_str = answer_str.lower()
                
                f_src.write(full_str + '\n')
                f_tgt.write(answer_str + '\n')
                history.append((question_str, answer_str))

                sanity_checks['full'] = full_str
                sanity_checks['answer'] = answer_str

    print(f'\nTraining Completed! There are', len(impossible_questions), 'impossible questions')
    print(impossible_questions)

    f_src.close()
    f_tgt.close()
