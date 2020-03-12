import argparse
import os
import re
import csv
import math


def find_result_file():
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for f in files:
        if f == "test_classifications.tsv":
            return f 
        elif f == "train_classifications.tsv":
            return f
    print("error - unable to find result file")

def find_output_id(result_file):
    dir_path = os.path.split(os.getcwd())[1]
    m = re.match('.*(test).*',dir_path)
    if m:
        test_or_train = "test"
    else:
        m = re.match('.*(train).*',dir_path)
        if m:
            test_or_train = "test"
        else:
            test_or_train = "unk"
    m = re.match('.*?([0-9]+)',dir_path)
    if m:
        number_id = m.group(1)
    else:
        number_id = 'xx'
    return "_"+test_or_train+"_"+number_id

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, default = None)
    parser.add_argument('--output_id', type=str, default = None)
    input_args = parser.parse_args()
    rf = input_args.result_file
    oid = input_args.output_id
    if rf == None:
        rf = find_result_file()
    if oid == None:
        oid = find_output_id(rf)
    
    num_true_pos = 0
    num_false_pos = 0
    num_true_neg = 0
    num_false_neg = 0

    with open(rf) as csv_file:
        csv_reader = csv.DictReader(csv_file,delimiter="\t")
        for row in csv_reader:
            classifier_judgement = row["classifier_judgement"]
            classifier_judgement = int(classifier_judgement)
            true_label = row["true_label"]
            true_label = int(true_label)
            if classifier_judgement == -1:
                print("WARNING: found classifier_judgement of -1. Will treat as 0 (rounding issue?)")
                print("\t for line with sentence "+row["sentence_1"])
                classifier_judgement = 0
            if true_label == -1:
                print("WARNING: found true_lable of -1. Will treat as 0 (rounding issue?)")
                print("\t for line with sentence "+row["sentence_1"])
                true_label = 0
            if classifier_judgement == 2:
                print("WARNING: found classifier_judgement of 2. Will treat as 1 (rounding issue?)")
                print("\t for line with sentence "+row["sentence_1"])
                classifier_judgement = 1
            if true_label == 2:
                print("WARNING: found true_lable of 2. Will treat as 1 (rounding issue?)")
                print("\t for line with sentence "+row["sentence_1"])
                true_label = 1
            if classifier_judgement == 0:
                if true_label == 0:
                    num_true_neg +=1
                elif true_label == 1:
                    num_false_neg +=1
                else:
                    print("ERROR! unknown value for true_label (should be 1 or 0)")
                    print("\t for line with sentence "+row["sentence_1"])
            elif classifier_judgement == 1:
                if true_label == 0:
                    num_false_pos +=1
                elif true_label == 1:
                    num_true_pos +=1
                else:
                    print("ERROR! unknown value for true_label (should be 1 or 0)")
                    print("\t for line with sentence "+row["sentence_1"])
            else:
                print("ERROR! unknown value for classifier_judgement (should be 1 or 0)")
                print("\t for line with sentence "+row["sentence_1"])
    recall = num_true_pos / (num_true_pos + num_false_pos)
    precision = num_true_pos / (num_true_pos + num_true_neg)
    accuracy = (num_true_pos + num_true_neg) / (num_true_pos + num_true_neg + num_false_pos + num_false_neg)
    f1 = 2*(precision*recall) / (precision+recall)
    with open("summary"+oid,'w') as fo:
        fo.write("accuracy = "+str(accuracy))
        fo.write("\n")
        fo.write("f1 = "+str(f1))
        fo.write("\n")
        fo.write("precision = "+str(precision))
        fo.write("\n")
        fo.write("recall = "+str(recall))
        fo.write("\n")
        fo.write("num_true_pos = "+str(num_true_pos))
        fo.write("\n")
        fo.write("num_false_pos = "+str(num_false_pos))
        fo.write("\n")
        fo.write("num_true_neg = "+str(num_true_neg))
        fo.write("\n")
        fo.write("num_false_neg = "+str(num_false_neg))
    
