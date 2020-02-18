import sys
import json

def load_idioms(filename):
    idioms = set()
    with open(filename,'r') as f:
        f.readline()
        for l in f:
            l = l.strip().split('\t')
            if int(l[1])  == 1:
                idioms.add(l[0])
    return idioms

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: get_comments_with_idiom.py <idiom_input_path> <comment_input_path> <output_path> ')
        exit()
    idioms = load_idioms(sys.argv[1])
    print("{} idioms found".format(len(idioms)))
    count = 0
    with open(sys.argv[2],'r') as f:
        with open(sys.argv[3], 'w') as w:
            w.write("Idiom\tSentence With Idiom\tFull Comment\n")
            for l in f:
                j = json.loads(l.strip())['body'].strip().replace("\n", "")
                found = 0
                for i in idioms:
                    if i in j and found == 0:
                        for a in j.split('.'):
                            if i in a:
                                found = 1
                                w.write("{}\t{}\t{}\n".format(i,a,j))
                                count += 1
                                break
    print('There were {} idioms found'.format(count))