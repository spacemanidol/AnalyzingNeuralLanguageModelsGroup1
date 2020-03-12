import crawl_the_dict
import random

def replacement_dict(samples):
    replacements = {}

    for sample in samples:
        idiom = sample[0]
        orig_sent = sample[1]
        replaced = sample[3]
        index_start = orig_sent.find(idiom)
        prefix_orig = orig_sent[:index_start]

        index_end = index_start + len(idiom)

        rpl_i_start_pref = replaced.find(prefix_orig)
        rpl_i_end_pref = rpl_i_start_pref + len(prefix_orig)


        if not index_end == len(orig_sent):
            suffix_orig = orig_sent[index_end:]
            rpl_i_start_suff = replaced.find(suffix_orig)
            replacement = replaced[rpl_i_end_pref:rpl_i_start_suff]

        else:
            replacement = replaced[rpl_i_end_pref:]

        # rpl_i_end_suff = rpl_i_start_suff + len(prefix_orig)
        # rpl_prefix = replaced[:rpl_i_end_pref]
        # print(rpl_prefix)

        if idiom not in replacements.keys():
            replacements[idiom] = []
            replacements[idiom].append(replacement)
        else:
            if not len(replacements[idiom]) == 4:
                replacements[idiom].append(replacement)
            else:
                continue

    return replacements


def load_comment_idioms(filename):
    samples = []
    with open(filename,'r', encoding="utf-8") as f:

        for line in f.readlines():
            l = line.strip().replace('\n', "").split('\t')

            if not len(l) == 5:
                print(l)
                print(line)

            samples.append((l[0].strip(),l[1].strip(),l[2].strip(), l[3].strip(), l[4].strip()))
    return samples


def write_rpl_dict(replacements, file):
    with open(file, "w", encoding="utf-8") as fw:
        for idiom, lst in replacements.items():
            fw.write(idiom+"\t" + "\t".join(lst) +"\n")
    fw.close()
    return

def read_rpl_file(file):
    replacements = {}

    with open(file, "r", encoding="utf-8") as fr:
        l = fr.readline()

        while l:
            items = l.strip().split("\t")
            idiom = items[0]
            replacements[idiom] = []
            replacements[idiom].append(items[1])
            replacements[idiom].append(items[2])
            replacements[idiom].append(items[3])
            replacements[idiom].append(items[4])

            l = fr.readline()

    return replacements


def propagate_replacements(input, output, rpl_dict):
        samples = crawl_the_dict.load_comment_idioms(input)
        rpl_count = 0
        with open(output, 'w', encoding="utf-8") as fw:
            for sample in samples:
                if sample[0] in rpl_dict.keys():
                    rpl_count +=1
                    paraphrase1, paraphrase2, nonparaphrase1, nonparaphrase2 = rpl_dict[sample[0]]
                    fw.write("{}\t{}\t{}\t{}\t{}\n".format(sample[0], sample[1], sample[2],
                                                          sample[1].replace(sample[0], paraphrase1), 1))
                    fw.write("{}\t{}\t{}\t{}\t{}\n".format(sample[0], sample[1], sample[2],
                                                          sample[1].replace(sample[0], paraphrase2), 1))
                    fw.write("{}\t{}\t{}\t{}\t{}\n".format(sample[0], sample[1], sample[2],
                                                          sample[1].replace(sample[0], nonparaphrase1), 0))
                    fw.write("{}\t{}\t{}\t{}\t{}\n".format(sample[0], sample[1], sample[2],
                                                          sample[1].replace(sample[0], nonparaphrase2), 0))
                else:
                    continue
        fw.close()
        print("Found {} replacements in {} samples".format(rpl_count, len(samples)))
        return

def rewrite_outputs(file, new_file, flag = 0, i=0):
    samples = load_comment_idioms(file)
    # print(samples)

    with open(new_file, "a", encoding="utf-8") as w:
        w.write("Index\tPara_val\tFlag\tIdiom\tSentence\tParaphrase\n")
        for j in range(len(samples)):

            if j % 4 != 0:
                w.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(str(i), samples[j][4], flag, samples[j][0], samples[j][1], samples[j][3]))
            else:
                i += 1
                w.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(str(i), samples[j][4], flag, samples[j][0], samples[j][1], samples[j][3]))
    return i

def load_idioms(file):
    samples = []
    with open(file, 'r', encoding="utf-8") as f:

        for l in f.readlines():
            l = l.strip().replace('\n', '').split('\t')

            if not len(l) == 6:
                print(l, len(l))

            samples.append((l[0].strip(), l[1].strip(), l[2].strip(), l[3].strip(), l[4].strip(), l[5].strip()))
    return samples

def shuffle_results(file, output):
    samples = load_idioms(file)[1:]
    random.shuffle(samples)

    with open(output, "w", encoding="utf-8") as w:
        w.write("Index\tPara_val\tFlag\tIdiom\tSentence\tParaphrase\n")
        for sample in samples:
            w.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(sample[0], sample[1], sample[2], sample[3], sample[4], sample[5]))
    w.close()
    return

def random_split(train_size, file, trainfile, devfile):
    samples = load_idioms(file)[1:]
    index = int(train_size*len(samples))
    train = samples[:index]
    dev = samples[index:]
    with open(trainfile, "w", encoding="utf-8") as trf:
        trf.write("Para_val\tIndex\tFlag\tSentence\tParaphrase\tIdiom\n")

        for sample in train:
            trf.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(sample[1], sample[0], sample[2], sample[4], sample[5], sample[3]))

    with open(devfile, "w", encoding="utf-8") as dvf:
        dvf.write("Para_val\tIndex\tFlag\tSentence\tParaphrase\tIdiom\n")

        for sample in dev:
            dvf.write(
                "{}\t{}\t{}\t{}\t{}\t{}\n".format(sample[1], sample[0], sample[2], sample[4], sample[5], sample[3]))

def stats_idioms(samples):
    idioms = {}
    for sample in samples:
        idiom = sample[3]
        if idiom in idioms.keys():
            idioms[idiom] +=1
        else:
            idioms[idiom] = 1
    return idioms

def read_idiom_stats(file):
    result = {}
    with open(file, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            unit = line.split("\t")
            result[unit[0].strip()] = int(unit[1].strip())
    return result

def get_split(infile, devfile, trainfile, dev_idioms_file):
    dev_idiom = read_idiom_stats(dev_idioms_file)
    samples = load_idioms(infile)[1:]

    with open(trainfile, "w", encoding="utf-8") as trf:
        trf.write("Para_val\tIndex\tFlag\tSentence\tParaphrase\tIdiom\n")
        with open(devfile, "w", encoding="utf-8") as dvf:
            dvf.write("Para_val\tIndex\tFlag\tSentence\tParaphrase\tIdiom\n")

            for sample in samples:
                if sample[3] in dev_idiom:
                    dvf.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(sample[1], sample[0], sample[2], sample[4], sample[5], sample[3]))
                else:
                    trf.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(sample[1], sample[0], sample[2], sample[4], sample[5], sample[3]))



if __name__ == '__main__':
    # samples = load_comment_idioms("daniel2output.tsv")
    # replacement_dict = replacement_dict(samples)
    # # print(replacement_dict)
    #
    # write_rpl_dict(replacement_dict, "daniel_2_rpl.tsv")
    # # rpl_d = read_rpl_file("daniel1output_1_rpl.tsv")
    #
    # i = rewrite_outputs("paraphrased_1.tsv", "joined_elena.tsv", flag=0, i=0)
    # # k = rewrite_outputs("daniel1out.tsv", "joined_all.tsv", flag=1, i=i)
    # # l = rewrite_outputs("daniel2out.tsv", "joined_all.tsv", flag=1, i = k)
    # m = rewrite_outputs("replacedelena2400.tsv", "joined_elena.tsv", flag=0, i = i)

    shuffle_results("joined_elena.tsv", "joined_and_shuffled_elena.tsv")

    random_split(0.8, "joined_and_shuffled_elena.tsv", "train_elena.tsv", "dev_elena.tsv")

    # propagate_replacements("daniel3.tsv", "process_daniel3.tsv", rpl_d)


    # # print(replacement_dict)
    # samples = load_idioms("joined_and_shuffled_all.tsv")[1:]
    # idioms = stats_idioms(samples)
    # idioms = sorted(idioms.items(), key=lambda x: x[1], reverse=True)
    # for idiom in idioms:
    #     print(idiom[0] + "\t" + str(idiom[1]))


    # get_split("joined_and_shuffled_all.tsv", "dev_unique.tsv", "train_unique.tsv", "dev_idioms")










