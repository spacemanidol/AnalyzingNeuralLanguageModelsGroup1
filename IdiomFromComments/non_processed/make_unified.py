with open('all.tsv','w') as w:
    #Quality #1 ID   #2 ID   #1 String       #2 String target format
    with open('2K_joined_and_shuffled.tsv','r') as f:
        #264     moment in the sun       My moment in the sun has arrived!       My moment of failure is now     0
        for l in f:
            l = l.strip().split('\t')
            q = l[4]
            s1 = l[2]
            s2 = l[3]
            w.write("{}\tNULL\tNULL\t{}\t{}\n".format(q, s1, s2))
    with open('daniel_idioms.tsv','r') as f:
        #slap in the face	Must be a slap in the face to those who complain about suggestions xD	Must be a slap in the face to those who complain about suggestions xD	Must be quite insulting to those who complain about suggestions xD	1
        for l in f:
            l = l.strip().split('\t')
            q = l[4]
            s1 = l[1]
            s2 = l[3]
            w.write("{}\tNULL\tNULL\t{}\t{}\n".format(q, s1, s2))


with open('dev.tsv','w') as w:
    w.write("Quality #1\tID   #2\tID   #1\tString       #2\tString target\n")


with open('train.tsv','w') as w:
    w.write("Quality #1\tID   #2\tID   #1\tString       #2\tString target\n")
