#!/bin/sh

for d in $(find . -mindepth 1 -type d )
do
echo run script for $d 
cp build_summary.py $d
cd $d
python build_summary.py
cd ..
done
