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

    if len(answers_list) == 1: # Check if it is the ONLY answer available
        pass
    elif answers_list[0][1] > answers_list[1][1]:  # Check if the first answer has the majority votes
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
    parser.add_argument('--pred_file', '-p', type=str, required=True)
    parser.add_argument('--output_file', '-o', type=str, required=True)
    args = parser.parse_args()

    with open(args.data_file) as f:
        dataset = json.load(f)

    question_ids = []
    # for i, datum in enumerate(dataset['data']):
    #     print('processing {}/{}...'.format(i, len(dataset['data'])))
    #     for question, answer in zip(datum['questions'], datum['answers']):
    #         assert question['turn_id'] == answer['turn_id']
    #         output.append({'id': datum['id'], 'turn_id': question['turn_id']})
    for i, datum in enumerate(dataset['data']):
        for paragraph in datum['paragraphs']:
            for qa in paragraph['qas']:

                # Retrieve question id
                question_id = qa['id']
                question_ids.append(question_id)
                # Simple Heuristic To Determine Answers
                # output.append({'id':question_id,'answer':answer_str})

    predictions = []
    with open(args.pred_file) as f:
        for line in f.readlines():
            predictions.append(line.strip())

    print(len(question_ids),len(predictions))
    assert len(question_ids) == len(predictions)
    output = dict(zip(question_ids, predictions))

    with open(args.output_file, 'w') as outfile:
        json.dump(output, outfile, indent=4)
