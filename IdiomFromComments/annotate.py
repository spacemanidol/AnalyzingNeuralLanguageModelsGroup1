import sys
import json

def load_comment_idioms(filename):
    samples = []
    with open(filename,'r') as f:
        f.readline()
        for l in f:
            l = l.strip().replace('\n','').split('\t')
            if len(l) == 3:
                samples.append((l[0].strip(),l[1].strip(),l[2].strip()))
    return samples

def create_annotation(idiom, comment, context):
    print("The idiom is:\n{}\n\nThe Sentence containing idiom is:\n{}\n\nThe fullComment is:\n{}\n\nPlease provide 2 paraphrases and 2 incorrect paraphrases".format(sample[0],sample[1],sample[2]))
    satisfied = 'n'
    while satisfied != 'y':
        if satisfied == 's':
            return 0,0,0,0
        paraphrase1 = input("Write your First Paraphrase\n")
        satisfied = input("You wrote\n{}\nAre you satisfied? Type y to proceedand s to skip current example\n".format(paraphrase1))
    satisfied = 'n'
    while satisfied != 'y':
        if satisfied == 's':
            return 0,0,0,0
        paraphrase2 = input("Write your Second Paraphrase\n")
        satisfied = input("You wrote\n{}\nAre you satisfied? Type y to proceed and s to skip current example\n".format(paraphrase2))
    satisfied = 'n'
    while satisfied != 'y':
        if satisfied == 's':
            return 0,0,0,0
        nonparaphrase1 = input("Write your First Non Paraphrase\n")
        satisfied = input("You wrote\n{}\nAre you satisfied? Type y to proceed and s to skip current example\n".format(nonparaphrase1))
    satisfied = 'n'
    while satisfied != 'y':
        if satisfied == 's':
            return 0,0,0,0
        nonparaphrase2 = input("Write your Second Non Paraphrase\n")
        satisfied = input("You wrote:\n{}\nAre you satisfied? Type y to proceed and s to skip current example\n".format(nonparaphrase2))
    return paraphrase1, paraphrase2, nonparaphrase1, nonparaphrase2
    

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: annotate.py <input_file> <output> <target_samples>')
        exit()
    samples = load_comment_idioms(sys.argv[1])
    print("{} samples found".format(len(samples)))
    count = 0
    target_count = int(sys.argv[3])
    with open(sys.argv[2],'w') as w:
        w.write("Idiom\tContextSentence\tFullComment\tTextParaphrase\tParaphraseValue\n") # for Paraphrase Value 0 = non paraphrase, 1 = paraphrase
        for sample in samples:
            if count == target_count:
                print("You have reached your goal to create {} samples. SICK!".format(target_count))
                break
            print("You have written {} samples".format(count))
            paraphrase1, paraphrase2, nonparaphrase1, nonparaphrase2 = create_annotation(sample[0],sample[1],sample[2])
            if paraphrase1 != 0:
                count += 1
                w.write("{}\t{}\t{}\t{}\t{}\n".format(sample[0],sample[1],sample[2],paraphrase1,1))
                w.write("{}\t{}\t{}\t{}\t{}\n".format(sample[0],sample[1],sample[2],paraphrase2,1))
                w.write("{}\t{}\t{}\t{}\t{}\n".format(sample[0],sample[1],sample[2],nonparaphrase1,0))
                w.write("{}\t{}\t{}\t{}\t{}\n".format(sample[0],sample[1],sample[2],nonparaphrase2,0))
            else:
                print("Example Skipped, Moving on!!!\n\n\n\n".upper())
