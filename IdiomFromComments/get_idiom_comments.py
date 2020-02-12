import sys
import json

def load_idioms(filename, min_idiom_length):
    idioms = set()
    with open(filename,'r') as f:
        f.readline()
        for l in f:
            l = l.strip().split('\t')
            if len(l[0].split()) > min_idiom_length:
                idioms.add(l[0])
    return idioms

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: get_comments_with_idiom.py <idiom_input_path> <comment_input_path> <output_path> <min_length>')
        exit()
    min_idiom_length = int(sys.argv[4])
    idioms = load_idioms(sys.argv[1],min_idiom_length)
    print("{} idioms with length > {} found".format(len(idioms), min_idiom_length))
    with open(sys.argv[2],'r') as f:
        with open(sys.argv[3], 'w') as w:
            w.write("Comment\tidiom\n")
            for l in f:
                j = json.loads(l.strip())['body']
                for i in idioms:
                    if i in j:
                        w.write("{}\t{}\n".format(j,i))
                        break