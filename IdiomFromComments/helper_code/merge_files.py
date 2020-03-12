def load_comment_idioms(filename):
    samples = []
    with open(filename,'r', encoding="utf-8") as f:

        for l in f.readlines()[1:]:
            l = l.strip().replace('\n', '').split('\t')

            if not len(l) == 5:
                print(l)

            samples.append((l[0].strip(),l[1].strip(),l[2].strip(), l[3].strip(), l[4].strip()))
    return samples

def sort_daniel(sample1, sample2):
    samples = sample1 + sample2
    groups = {}
    for sample in samples:
        if sample[3] in groups.keys():
            groups[sample[3]].append(sample)
        else:
            groups[sample[3]] = []
            groups[sample[3]].append(sample)
    return groups

def rewrite_outputs(file, new_file, i=0):
        samples = load_comment_idioms(file)

        with open(new_file, "a", encoding="utf-8") as w:
            for j in range(len(samples)):

                if j % 4 != 0:
                    w.write("{}\t{}\t{}\t{}\t{}\n".format(str(i), samples[j][0], samples[j][1], samples[j][3],
                                                          samples[j][4]))
                else:
                    i += 1
                    w.write("{}\t{}\t{}\t{}\t{}\n".format(str(i), samples[j][0], samples[j][1], samples[j][3],
                                                          samples[j][4]))
        return i



if __name__ == '__main__':
    file1 = "dev.tsv"
    file2 = "train.tsv"

    sample1 = load_comment_idioms(file1)
    sample2 = load_comment_idioms(file2)

    print(sort_daniel(sample1, sample2))
