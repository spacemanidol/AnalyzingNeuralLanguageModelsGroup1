import sys
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: modify_format.py <input_path> <output_path>')
        exit()
    with open(sys.argv[1],'r') as f:
        with open(sys.argv[2], 'w') as w:
            w.write('sentence\tlabel\n')
            for l in f:
                l = l.strip().split('\t')
                if len(l) == 4:
                    w.write("{}\t{}\n".format(l[3], l[1]))
