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
        print('Usage: get_idiom_msmarco.py <idiom_input_path> <comment_input_path> <output_path> <min_length>')
        exit()
    min_idiom_length = int(sys.argv[4])
    idioms = load_idioms(sys.argv[1],min_idiom_length)
    print("{} idioms with length > {} found".format(len(idioms), min_idiom_length))
    idioms_found = 0
    with open(sys.argv[2],'r') as f:
        with open(sys.argv[3], 'w') as w:
            w.write("Query\tidiom\n")
            for l in f:
                j = l.strip().split('\t')[1]
                for i in idioms:
                    if i in j and 'meaning' not in j and 'define' not in j and 'mean' not in j and 'said' not in j and 'sings' not in j and 'sing' not in j:
                        w.write("{}\t{}\n".format(j,i))
                        idioms_found += 1
                        break
    print("{} Idioms found in {}".format(idioms_found, sys.argv[2]))
