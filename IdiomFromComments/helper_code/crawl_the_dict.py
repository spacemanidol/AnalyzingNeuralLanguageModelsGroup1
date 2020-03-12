import re
from nltk import pos_tag
from nltk import word_tokenize
import requests
from bs4 import BeautifulSoup
from nltk.corpus import wordnet as wn
import sys

def load_comment_idioms(filename):
    samples = []
    with open(filename,'r', encoding="utf-8") as f:
        f.readline()
        for l in f:
            l = l.strip().replace('\n','').split('\t')
            if len(l) == 3:
                samples.append((l[0].strip(),l[1].strip(),l[2].strip()))
    return samples

def crawl_dict(idiom):
    '''
    Obtain the paraphrases from the lists of synonyms stored in the dictionary
    :param idiom: string
    :return: idiomatic paraphrase if available, one word paraphrase if not
    '''

    not_found = False
    article = ["a", "the", "(you)", "to", "something"]

    tokens = idiom.lower().split()
    idiom_suff = "-".join(tokens)
    link = 'https://www.macmillanthesaurus.com/' + idiom_suff
    page = requests.get(link)

    replacement_idiom = 0
    replacement_word = 0

    # if there is no entry for the idiom, try to modify it:
    if str(page) == "<Response [404]>":
        # try to vary get and come
        if "get" in tokens:
            index = idiom.find("get") + 3
            tokens = idiom[index:].split()
            idiom_suff = idiom[:index] + "-come-" + "-".join(tokens)
            idiom = idiom[:index] + "/come" + idiom[index:]
            link = 'https://www.macmillanthesaurus.com/' + idiom_suff
            page = requests.get(link)

        else:
            # try articles, "some", "you"
            i = 0
            while (str(page) == "<Response [404]>" and i < len(article)):

                if ("(" or ")") in article[i]:
                    idiom_suff = article[i][1:-1] + "-" + "-".join(tokens)
                    new_idiom = "\\" + article[i][:-1] + "\\)" + " " + idiom

                elif article[i] == "something":
                    idiom_suff = "-".join(tokens) + "-" + article[i]
                    new_idiom = idiom + " " + article[i]
                else:
                    idiom_suff = article[i] + "-" + "-".join(tokens)
                    new_idiom = article[i] + " " + idiom

                link = 'https://www.macmillanthesaurus.com/' + idiom_suff
                page = requests.get(link)

                i += 1

            idiom = new_idiom

            # if after the trial the idiom was not found

        if str(page) == "<Response [404]>":
            print("Idiom not found: {}".format(idiom))
            not_found = True
            return 0,0

    if not_found == False:
        print("Processing idiom {}".format(idiom))

        soup = BeautifulSoup(page.content, 'html.parser')
        # print(soup)
        search_string = "Synonyms for '{}': ".format(idiom)
        pattern = re.compile(r'(?<={0}).+(?=" name=\"description\")'.format(search_string))
        # print(pattern)
        syns = re.findall(pattern, str(soup))
        # print(syns)

        if len(syns) == 0:
            return 0, 0

        else:
            syn_list = syns[0].split(", ")

        idiom_syns = [syn for syn in syn_list if
                      (len(syn.split()) >= 3 and "(" in syn)]

        word_syns = [syn for syn in syn_list if not syn in idiom_syns]

        print("Found replacements")

        if len(idiom_syns) > 0 and len(word_syns) > 0:
            replacement_idiom = idiom_syns[0]
            replacement_word = word_syns[0]
            # print("Found 1")

        elif len(idiom_syns) == 0 and len(word_syns) > 1:
            replacement_idiom = word_syns[0]
            replacement_word = word_syns[1]
            # print("Found 2")


        elif len(word_syns) == 0 and len(idiom_syns) > 1:
            replacement_idiom = idiom_syns[0]
            replacement_word = idiom_syns[1]

            # print("Found 3")


        else:
            print("No replacements found")
            replacement_idiom = 0
            replacement_word = 0



    return replacement_idiom, replacement_word


def get_non_paraphrases(idiom):
    '''
    Obtain bad paraphrases (literal replacements) from wordnet
    :param idiom: string - idiom
    :return:
    '''
    nouns = ["NN", "NNP", "NNS"]
    verbs = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    tokens = word_tokenize(idiom)
    parse = pos_tag(tokens)
    hypernyms = {}
    hyponyms = {}

    result_hypers = {}
    result_hypos = {}


    for element in parse:
        if element[1] in nouns:
            w_senses = wn.synsets(element[0], pos='n')

        elif element[1] in verbs:
            w_senses = wn.synsets(element[0], pos='v')
        else:
            continue

        if len(w_senses) == 0:
            print("No senses found for idiom: {}".format(idiom))
            break
        else:
            i = 0
            hypers = w_senses[i].hypernyms()
            while len(hypers) == 0 and i < len(w_senses)-1:
                i+=1
                hypers = w_senses[i].hypernyms()

            if len(hypers) == 0:
                break

            else:

                hyper = w_senses[i].hypernyms()[0].name()
                hypernyms[element[0]] = hyper

            j = 0
            hypos = w_senses[j].hyponyms()

            while len(hypos) == 0 and j < len(w_senses)-1:
                j += 1
                hypos = w_senses[j].hyponyms()

            if len(hypos) == 0:
                break
            else:

                hypo = w_senses[j].hyponyms()[0].name()
                hyponyms[element[0]] = hypo

    if not len(hypernyms) == 0:

        for word, syn in hypernyms.items():
            index = syn.find(".")
            syn = syn[:index].replace("_", " ")
            result_hypers[word] = syn

    if not len(hyponyms) == 0:

        for word, syn in hyponyms.items():
            index = syn.find(".")
            syn = syn[:index].replace("_", " ")
            result_hypos[word] = syn

    nonpara1 = []
    for word in tokens:
        if word in result_hypers.keys():
            nonpara1.append(result_hypers[word])
        else:
            nonpara1.append(word)
    nonpara1 = " ".join(nonpara1)

    nonpara2 = []
    for word in tokens:
        if word in result_hypos.keys():
            nonpara2.append(result_hypos[word])
        else:
            nonpara2.append(word)
    nonpara2 = " ".join(nonpara2)

    return nonpara1, nonpara2

if __name__ == '__main__':

    samples = load_comment_idioms(sys.argv[1])
    print("{} samples found".format(len(samples)))
    target_count = int(sys.argv[4])


    done_count = 0
    failed_count = 0
    with open(sys.argv[2], 'a', encoding="utf-8") as w:
        with open(sys.argv[3], 'a', encoding="utf-8") as failed_p:
            w.write(
                "Idiom\tContextSentence\tFullComment\tTextParaphrase\tParaphraseValue\n")  # for Paraphrase Value 0 = non paraphrase, 1 = paraphrase
            for sample in samples:
                if done_count == target_count:
                    print("You have reached your goal to create {} samples. SICK!".format(target_count))
                    break

                paraphrase1, paraphrase2 = crawl_dict(sample[0])
                print("Got paraphrases")
                nonparaphrase1, nonparaphrase2 = get_non_paraphrases(sample[0])
                print("Got non-paraphrases")

                if paraphrase1 != 0 and paraphrase2 != 0:
                    done_count += 1
                    w.write("{}\t{}\t{}\t{}\t{}\n".format(sample[0], sample[1], sample[2], sample[1].replace(sample[0], paraphrase1), 1))
                    w.write("{}\t{}\t{}\t{}\t{}\n".format(sample[0], sample[1], sample[2], sample[1].replace(sample[0], paraphrase2), 1))
                    w.write("{}\t{}\t{}\t{}\t{}\n".format(sample[0], sample[1], sample[2], sample[1].replace(sample[0], nonparaphrase1), 0))
                    w.write("{}\t{}\t{}\t{}\t{}\n".format(sample[0], sample[1], sample[2], sample[1].replace(sample[0], nonparaphrase2), 0))


                else:

                    failed_count +=1
                    failed_p.write("{}\t{}\t{}\t{}\t{}\n".format(sample[0], sample[1], sample[2],
                                                          paraphrase1, 1))
                    failed_p.write("{}\t{}\t{}\t{}\t{}\n".format(sample[0], sample[1], sample[2],
                                                          paraphrase2, 1))
                    failed_p.write("{}\t{}\t{}\t{}\t{}\n".format(sample[0], sample[1], sample[2],
                                                          sample[1].replace(sample[0], nonparaphrase1), 0))
                    failed_p.write("{}\t{}\t{}\t{}\t{}\n".format(sample[0], sample[1], sample[2],
                                                          sample[1].replace(sample[0], nonparaphrase2), 0))

    print("Total: {}; Processed: {}, Paraphrased: {}; Failed: {}".format(len(samples), done_count+failed_count, done_count, failed_count))

