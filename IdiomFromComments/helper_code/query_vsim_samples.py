import re
import requests
from bs4 import BeautifulSoup
import sys

def read_idioms(file):
    idioms = []
    with open(file, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            line = line.strip().split("\t")
            idioms.append((line[0], line[1], line[2]))
    return idioms

def query_dict(word):

    link = 'https://www.ldoceonline.com/dictionary/' + word
    page = requests.get(link)

    soup = BeautifulSoup(page.content, 'html.parser')
    all_examples = []

    examples = soup.find_all("span", {"class": "cexa1g1 exa"})
    if len(examples) == 0:
        examples = soup.find_all("span", {"class": "cexa1 exa"})
    i = 1
    while examples:

        for example in examples:
            example = example.text.lstrip("• ")
            if len(example.split()) >= 5:
                all_examples.append(example)
        i+=1
        search = "cexa1g{} exa".format(i)
        examples = soup.find_all("span", {"class": search})


    return all_examples

def process_all(idioms_file):
    with open (idioms_file, "w", encoding="utf-8") as fw:
        fw.write("sentence_id\tpair_id\tsentence\tword\tfigurative\n")

        for i in range(len(idioms)):

            target = idioms[i][1]
            meaning = idioms[i][2]

            target_ex = query_dict(target)
            meaning_ex = query_dict(meaning)


            for ex in target_ex:
                fw.write("{}\t{}\t{}\t{}\t{}\n".format(0, i+1, ex, target, 0))
            for ex in meaning_ex:
                fw.write("{}\t{}\t{}\t{}\t{}\n".format(0, i+1, ex, meaning, 1))

            for x in range(1, 6):
                fw.write("{}\t{}\t{}\t{}\t{}\n".format(0, i+1, "", target, 1))
            print("Processed {} example".format(i))


def rewrite_file(infile, outfile):
    with open (infile, "r", encoding="utf-8") as fr:
        with open(outfile, "w", encoding='utf-8') as fw:
            fw.write("sentence_id\tpair_id\tsentence\tword\tfigurative\n")
            lines = fr.readlines()[1:]
            i = 0
            pair_id = 1
            figurative = 0
            z = 0
            while i < len(lines):
                table = lines[i].strip().split("\t")

                if i % 25 == 0 and i !=0:
                    pair_id += 1
                    figurative = 0
                    z = 0
                elif i == 0:
                    figurative = 0
                    z = 0
                else:
                    pair_id = pair_id
                    z+=1

                if z % 20 == 0 and z !=0:
                    figurative = 1
                print(i, z, lines[i])

                if i > 451:
                    figurative = table[4]

                fw.write("{}\t{}\t{}\t{}\t{}\n".format(i+1, pair_id, table[2], table[3], figurative))

                i += 1





                # if z == 1:
                # if (i % 21 == 0 or i%22 == 0 or i%23 == 0 or i%24==0 or i%25==0) and not i == 0:








if __name__ == '__main__':
    # idioms = read_idioms("idioms_for_vsim")
    rewrite_file("idioms_for_similarity.tsv", "output_similarity")

