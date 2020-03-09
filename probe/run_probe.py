import json
from torch.autograd import Variable
import torch
from load_data import ParaphraseDataset
import argparse
import logging
import time
import os.path
from scipy.spatial.distance import cosine
from statistics import mean 

train, test = 'train', 'test'
combined, cls, pool = 'combined', 'cls', 'cls_pool'
module_logger = logging.getLogger('probe')

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def train_probe(input_args):
    model_name = input_args.model if input_args.model is not None else 'model.pt'
    indices = tuple(int(x) for x in input_args.indices.split('.'))
    train_data = ParaphraseDataset(input_args.input, input_args.embedding_model, input_args.embedding_batch_size,
                                   input_args.run_name,  indices=indices)
    data_labels = train_data.get_labels()

    paraphrase_embeddings = get_embeddings(train_data, input_args)

    # https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
    learning_rate = input_args.learning_rate
    epochs = input_args.epochs
    logging_increment = epochs / 10

    model = LinearRegression(paraphrase_embeddings.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    min_loss_reduction = input_args.min_loss_step
    prev_loss = -1
    for epoch in range(epochs):
        inputs = Variable(paraphrase_embeddings)
        labels = Variable(data_labels)

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs, labels)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        if prev_loss != -1:
            if prev_loss - loss < min_loss_reduction:
                module_logger.info('epoch {}, loss {}'.format(epoch, loss.item()))
                module_logger.info('Minimum loss reduction not achieved, aborting')
                epochs = epoch
                break

        prev_loss = loss
        if epoch % logging_increment == 0:
            module_logger.info('epoch {}, loss {}'.format(epoch, loss.item()))

    eval_model(model, paraphrase_embeddings, labels, input_args, train_data)
    output_file(input_args.run_name, 'model_metadata.json', json.dumps({
        'learning_rate': learning_rate,
        'epochs': epochs,
        'seed': input_args.rand_seed,
        'training_file': input_args.input,
        'embedding_paradigm': input_args.embedding_paradigm

    }))
    model_loc = os.path.join('output', input_args.run_name, model_name)
    module_logger.info("Saving model as {}".format(model_loc))
    torch.save(model.state_dict(), model_loc)


def get_embeddings(data, input_args):
    if input_args.embedding_cache is None:
        if input_args.embedding_paradigm == combined:
            encoded_data = data.get_flattened_encoded()
        else:
            encoded_data = data.get_encoded()
        embeddings, inputs, indices, pools = data.bert_word_embeddings(encoded_data)
    else:
        embeddings, inputs, indices, pools = data.load_saved_embeddings(input_args.embedding_cache)

    if input_args.embedding_paradigm == combined:
        final_embeddings = data.combine_sentence_embeddings(data.aggregate_sentence_embeddings(embeddings, inputs,
                                                                                                   indices))
    elif input_args.embedding_paradigm == cls:
        final_embeddings = data.bert_cls_embeddings(embeddings)
    elif input_args.embedding_paradigm == pool:
        final_embeddings = pools
    else:
        raise Exception("Unknown embedding paradigm")

    return final_embeddings


def test_probe(input_args):
    indices = tuple(int(x) for x in input_args.indices.split('.'))
    test_data = ParaphraseDataset(input_args.input, input_args.embedding_model, input_args.embedding_batch_size,
                                  input_args.run_name, indices=indices)
    labels = test_data.get_labels()

    final_embeddings = get_embeddings(test_data, input_args)

    model = LinearRegression(final_embeddings.shape[1], 1)
    model.load_state_dict(torch.load(input_args.model))

    eval_model(model, final_embeddings, labels, input_args, test_data)


def eval_round(in_tensor):
    return (in_tensor >= 0.5) * 1


def eval_model(model, data, input_labels, input_args, dataset):
    raw_for_out = dataset.get_raw_for_output()
    flat_labels = input_labels.flatten()
    model.eval()
    with torch.no_grad():
        inputs = Variable(data)
        labels = Variable(flat_labels)

        outputs = model(inputs)

        predicted_outputs = torch.squeeze(outputs)

    total = len(predicted_outputs)

    correct = int(torch.sum((eval_round(predicted_outputs) == labels.bool()) * 1))

    output_lines = ['\t'.join(('classifier_prob', 'classifier_judgement') + raw_for_out[0])+'\n'] + [
        '\t'.join((str(float(x)), str(int(eval_round(x)))) + raw_for_out[index+1]) + '\n'
        for index, x in enumerate(predicted_outputs)
    ]

    stats_out(input_args, dataset, predicted_outputs, labels)

    acc_string = "{}/{} correct for an accuracy of {}".format(correct, total, correct/total)
    output_file(input_args.run_name, '{}_classifications.tsv'.format(input_args.run), output_lines)
    output_file(input_args.run_name, '{}_acc.txt'.format(input_args.run), [acc_string + '\n'])
    module_logger.info(acc_string)


def stats_out(input_args, dataset, predicted_outputs, labels):
    if input_args.embedding_paradigm == combined:
        all_results, summary = sentence_vector_sim_calculations(input_args, dataset, predicted_outputs)
        output_lines = format_for_output(summary) + ["{} \n".format(pair) for pair in all_results] 
        output_file(input_args.run_name, '{}_sent_vector_cosine_sim.txt'.format(input_args.run), output_lines)

    f1_stats = calculate_f1_results(predicted_outputs, labels)
    output_file(input_args.run_name, '{}_f1_stats.txt'.format(input_args.run), format_for_output(f1_stats))


def calculate_f1_results(predicted_outputs, labels):
    num_true_pos, num_true_neg, num_false_pos, num_false_neg = 0, 0, 0, 0
    
    for i, prediction in enumerate(predicted_outputs):
        if eval_round(prediction):
            if labels[i]:
                num_true_pos += 1
            else:
                num_false_pos += 1
        elif labels[i]:
            num_false_neg += 1
        else:
            num_true_neg += 1

    recall = num_true_pos / (num_true_pos + num_false_pos)
    precision = num_true_pos / (num_true_pos + num_true_neg)
    accuracy = (num_true_pos + num_true_neg) / (num_true_pos + num_true_neg + num_false_pos + num_false_neg)
    f1 = 2 * (precision*recall) / (precision+recall)

    return {
        "num_true_positive": num_true_pos,
        "num_false_positive": num_false_pos,
        "num_true_negative": num_true_neg,
        "num_false_negative": num_false_neg,
        "acaccuracy": accuracy,
        "f1": f1
    }


def sentence_vector_sim_calculations(input_args, dataset, predicted_outputs):
    if input_args.embedding_cache:
        embedding_cache_file = input_args.embedding_cache
    else:
        embedding_cache_file = "cache/"+input_args.run_name

    embeddings, inputs, indices, _pools = dataset.load_saved_embeddings(embedding_cache_file)
    sentence_embeddings = dataset.aggregate_sentence_embeddings(embeddings, inputs, indices)
    
    data = dataset.get_raw_for_output()[1:]
    cosine_comps = [calculate_paraphrase_pair_similarity(i, pair_sents, 
                                                            sentence_embeddings, predicted_outputs) 
                                for i, pair_sents in enumerate(data)]

    correctly_judged_paraphrases = [pair['cosine_similarity'] for pair in cosine_comps 
                                    if pair['label'] and pair['judgment']]
    correctly_judged_non_paraphrases = [pair['cosine_similarity'] for pair in cosine_comps 
                                        if not pair['label'] and not pair['judgment']]
    incorrectly_judged_paraphrases =  [pair['cosine_similarity'] for pair in cosine_comps 
                                        if pair['label'] and not pair['judgment']]
    incorrectly_judged_non_paraphrases =  [pair['cosine_similarity'] for pair in cosine_comps 
                                            if not pair['label'] and pair['judgment']]

    return cosine_comps, {
        'number correctly judged paraphrases': len(correctly_judged_paraphrases),
        'number incorrectly judged paraphrases': len(incorrectly_judged_paraphrases),
        'number correctly judged non paraphrases': len(correctly_judged_non_paraphrases),
        'number incorrectly judged non paraphrases': len(incorrectly_judged_non_paraphrases),
        'average_cosine_sim_for_correctly_judged_paraphrases': handle_zero_case(correctly_judged_paraphrases),
        'average_cosine_sim_for_correctly_judged_non_paraphrases': handle_zero_case(correctly_judged_non_paraphrases),
        'average_cosine_sim_for_incorrectly_judged_paraphrases': handle_zero_case(incorrectly_judged_paraphrases),
        'average_cosine_sim_for_incorrectly_judged_non_paraphrases': handle_zero_case(incorrectly_judged_non_paraphrases),
        'average_cosine_for_paraphrases': handle_zero_case(correctly_judged_paraphrases + incorrectly_judged_paraphrases),
        'average_cosine_for_non_paraphrases': handle_zero_case(correctly_judged_non_paraphrases + incorrectly_judged_non_paraphrases)
    }


def calculate_paraphrase_pair_similarity(index, raw_data, sentence_embeddings, predicted_outputs):
    cosine_sim = 1 - cosine(sentence_embeddings[index][0], sentence_embeddings[index][1])
    try:
        idiom = raw_data[5]
    except:
        idiom = "N/A"

    return {
        'dataset_index': index,
        #"idiom": raw_data[5],
        "idiom": idiom ,
        'label': int(raw_data[0]),
        'judgment': bool(eval_round(predicted_outputs[index])),
        'sent_1': raw_data[1],
        'sent_2': raw_data[2],
        'index': raw_data[3],
        'cosine_similarity': cosine_sim
    }    


def handle_zero_case(category_results):
    if not category_results:
        return 'N/A'
    return mean(category_results)


def format_for_output(metric_dict):
    return ["{}: {}\n".format(k, v) for k, v in metric_dict.items()] + ["\n"]


def output_file(run_name, filename, content):
    folder = os.path.join('output', run_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    module_logger.info('Writing {} to {}'.format(filename, folder))
    with open(os.path.join(folder, filename), 'w+') as outfile:
        outfile.writelines(content)


def require_args(input_args, required_args):
    for required_arg in required_args:
        if getattr(input_args, required_arg) is None:
            raise Exception("Cannot run {} without --{}".format(input_args.run, required_arg))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_batch_size', type=int, default=64)
    parser.add_argument('--embedding_cache', type=str, help='Directory to load cached embeddings from')
    parser.add_argument('--embedding_model', type=str, default='bert-large-uncased',
                        help='The model used to transform text into word embeddings')
    parser.add_argument('--embedding_paradigm', type=str, choices=[combined, cls, pool], default=combined,
                        help='Whether to combine sentence embeddings or take the CLS token of joint embeddings')
    parser.add_argument('--run', type=str, choices=[train, test], required=True)
    parser.add_argument('--input', type=str, required=True)

    # mrpc indices = 0.3.4
    # our dataset indices = 0.3.4
    parser.add_argument('--indices', type=str, default='0.3.4')

    parser.add_argument('--run_name', type=str, default='run_{}'.format((int(time.time()))),
                        help='A label for the run, used to name output and cache directories')

    parser.add_argument('--model', type=str, help='Name of the model')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training', default=0.001)
    parser.add_argument('--epochs', type=int, help='Epochs for training', default=1000)
    parser.add_argument('--min_loss_step', type=float, default=0.0001,
                        help='Minimum epoch loss; smaller improvements than this will cause training to abort')
    parser.add_argument('--rand_seed', type=int, default=0)

    input_args = parser.parse_args()
    torch.manual_seed(input_args.rand_seed)

    if input_args.run == train:
        train_probe(input_args)
    elif input_args.run ==test:
        require_args(input_args, ['model'])
        test_probe(input_args)




