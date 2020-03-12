def count(all_data, sampled):
    # Index - 0
    # Para_val -1
    # Flag -2
    # Idiom - 3
    # Sentence -4
    # Paraphrase -5
    total_all = 0
    total_sub = 0
    intersect = 0
    samples = []
    with open(all_data, "r", encoding="utf-8") as fr:
        for line in fr.readlines()[1:]:
            sample = line.strip().split("\t")
            samples.append(sample[4])
            total_all += 1
    print(samples)

    with open(sampled, "r", encoding="utf-8") as sub:
        for line in sub.readlines()[1:]:
            line = line.strip().split("\t")
            total_sub +=1
            if line[4] in samples:
                print(line)
                intersect +=1

    return intersect, total_all, intersect/total_all, intersect/total_sub

if __name__ == '__main__':
    file1 = "train_all.tsv"
    file2 = "dev.tsv"
    res = count(file1, file2)

    print("Number of intersecting samples in {0} and {1}: {2}; \nPercentage of {3}: {4} \nPercentage of {5}: {6}".format(file1, file2, res[0],
                                                                                                                         file1, res[2], file2, res[3]))