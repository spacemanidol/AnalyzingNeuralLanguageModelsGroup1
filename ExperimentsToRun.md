Experiments
We broadly have a few pools of dev and train to sample from

TrainMRPC is 4076 and sampled at random. Call T1. Produces a downsampled 1600 sample called T1, and a downsampled 800 sample called T1,,
TrainIdiomRandom is 1600 and sampled at random. Call T2. Produces a downsampled 800 sample  called T2,,
TrainIdiomSorted is 1600 and sampled so none of its idioms are in its dev. Call T3. Produces a downsampled 800 sample called T3,
TrainIdiomDaniel is 800 and sampled at random. Call T4
TrainIdiomElena is 800 and sampled at random. Call T5

DevMRPC is 1726 and sampled at random. Call D1. Produces a downsampled 400 sample called D1, and a downsampled 200 sample called D1,,
DevIdiomRandom is 400 and sampled at random. Call D2. Produces a downsampled 200 sample  called D2,
DevIdiomSorted is 400 and sampled so none of its idioms are in its Train. Call D3. Produces a downsampled 200 sample called D3,
DevIdiomDaniel is 200 and sampled at random. Call D4
DevIdiomElena is D5 and sampled at random. Call D5


Experiment to test effect of training set size
1. Train and test on full MRPC(4076 Train 1726 Dev )  T1 D1
2. Train and test on MRPC (1600 Train 400 Dev ) T1, D1,
3. Train and test on MRPC (800 Train 200 Dev) T1,, D1,,

4. Train and Test on all our data(random sample) (1600 Train 400 Dev)  T2 D2
5. Train and Test on our data(random sample) (800 Train 200 Dev) T2, D2,

6. Train and Test on our data(idioms in Dev not in train)(1600 Train 400 Dev)  T3 D3
7. rain and Test on our data(idioms in Dev not in train)(800 Train 200 Dev)  T3, D3,

Experiment to test performance across samples
7. Train T1, test on D2
8. Train on T1, test on D3
9. Train on T1,, Test on D2,
10. Train on T1,, Test on D3,
11. Train on T1,, Test on D4
12. Train on T1,, Test on D5

Experiments on Agreement cross data
13. Train on T2, test on D1,,
14. Train on T3, test on D1,,
15. Train on T4 test on D5
16. Train on T4 test on D1,,
17. Train on T5 test on D4
18. Train on T5 test on D1,,