import csv

with open('data_table.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        bash_output_file = "run_trial_"+row["Trial"]+".sh"
        with open(bash_output_file,'w') as fo:
            fo.write("#!/bin/bash\n")
            fo.write("source ~/.bashrc\n")
            fo.write("conda activate my-project\n\n")
            fo.write("#Run trial %s train\n"%row["Trial"])

            fo.write(row["Command for training"])
            fo.write("\n\nwait\n")
            fo.write("#Run trial %s test\n"%row["Trial"])
            fo.write(row["Command for testing"]+"\n")
        condor_output_file = "condor_run_trial_"+row["Trial"]+".cmd"
        with open(condor_output_file,'w') as fo:
            fo.write("universe = vanilla\nexecutable="+bash_output_file+"\n")
            fo.write("getenv = true\noutput = run_trial_"+row["Trial"]+".out\n")
            fo.write("error = run_trial_"+row["Trial"]+".err\n")
            fo.write("log = run_trial_"+row["Trial"]+".log\n")
            fo.write("notification = complete\narguments = \ntransfer_executable = false\n+Research = True\nqueue\n")
'''
bashCommand = "chmod 777 *.sh"
import subprocess
process = subprocess.run(bashCommand)
'''

