
def get_selected(file):
    with open(file, "r", encoding="utf-8") as f:
        with open("selected_idioms", "w", encoding="utf-8") as fw:
            for line in f.readlines():
                sample = line.split("\t")

                if int(sample[1]) == 1:
                    fw.write(line)

if __name__ == '__main__':
    get_selected("idioms+3.tsv")