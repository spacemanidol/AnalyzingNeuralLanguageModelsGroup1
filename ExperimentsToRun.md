# Experiments  
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
  
  
# Effects of training data size on task
Exp1 Train and test on full MRPC(4076 Train 1726 Dev ) T1 D1  
{"eval_acc": 0.7507246376811594, "eval_f1": 0.8299050632911392, "eval_acc_and_f1": 0.7903148504861492, "learning_rate": 4.705882352941177e-05, "loss": 0.6097050496935844, "step": 100}  
{"eval_acc": 0.8057971014492754, "eval_f1": 0.8491670418730302, "eval_acc_and_f1": 0.8274820716611528, "learning_rate": 4.411764705882353e-05, "loss": 0.45668196216225626, "step": 200}  
{"eval_acc": 0.8318840579710145, "eval_f1": 0.8729184925503943, "eval_acc_and_f1": 0.8524012752607044, "learning_rate": 4.11764705882353e-05, "loss": 0.3154626343399286, "step": 300}  
{"eval_acc": 0.8127536231884058, "eval_f1": 0.8689655172413792, "eval_acc_and_f1": 0.8408595702148924, "learning_rate": 3.8235294117647055e-05, "loss": 0.2199432573001832, "step": 400}  
{"eval_acc": 0.8278260869565217, "eval_f1": 0.8751576292559899, "eval_acc_and_f1": 0.8514918581062558, "learning_rate": 3.529411764705883e-05, "loss": 0.1373146472685039, "step": 500}  
{"eval_acc": 0.8342028985507246, "eval_f1": 0.8811305070656691, "eval_acc_and_f1": 0.8576667028081968, "learning_rate": 3.235294117647059e-05, "loss": 0.09981583103071898, "step": 600}  
{"eval_acc": 0.8185507246376812, "eval_f1": 0.8736374646750101, "eval_acc_and_f1": 0.8460940946563457, "learning_rate": 2.9411764705882354e-05, "loss": 0.07589328840957023, "step": 700}  
{"eval_acc": 0.8318840579710145, "eval_f1": 0.8769100169779286, "eval_acc_and_f1": 0.8543970374744716, "learning_rate": 2.647058823529412e-05, "loss": 0.05558009307191242, "step": 800}█  
{"eval_acc": 0.8127536231884058, "eval_f1": 0.8650229837024654, "eval_acc_and_f1": 0.8388883034454356, "learning_rate": 2.3529411764705884e-05, "loss": 0.026701140822551678, "step": 900}  
{"eval_acc": 0.8260869565217391, "eval_f1": 0.8756218905472636, "eval_acc_and_f1": 0.8508544235345014, "learning_rate": 2.058823529411765e-05, "loss": 0.017476224560814446, "step": 1000}  
{"eval_acc": 0.8202898550724638, "eval_f1": 0.8716887417218544, "eval_acc_and_f1": 0.8459892983971591, "learning_rate": 1.7647058823529414e-05, "loss": 0.014409473883133614, "step": 1100}  
{"eval_acc": 0.8231884057971014, "eval_f1": 0.8691548691548692, "eval_acc_and_f1": 0.8461716374759853, "learning_rate": 1.4705882352941177e-05, "loss": 0.01835168295074254, "step": 1200}█  
{"eval_acc": 0.8243478260869566, "eval_f1": 0.873697373905794, "eval_acc_and_f1": 0.8490225999963753, "learning_rate": 1.1764705882352942e-05, "loss": 0.006331848980335053, "step": 1300}  
{"eval_acc": 0.8168115942028985, "eval_f1": 0.8699588477366256, "eval_acc_and_f1": 0.8433852209697621, "learning_rate": 8.823529411764707e-06, "loss": 0.01216069263376994, "step": 1400}  
{"eval_acc": 0.8220289855072463, "eval_f1": 0.8712788259958071, "eval_acc_and_f1": 0.8466539057515268, "learning_rate": 5.882352941176471e-06, "loss": 0.008074100170197197, "step": 1500}  
{"eval_acc": 0.8249275362318841, "eval_f1": 0.8727885425442291, "eval_acc_and_f1": 0.8488580393880566, "learning_rate": 2.9411764705882355e-06, "loss": 0.000751012740074657, "step": 1600}  
{"eval_acc": 0.8243478260869566, "eval_f1": 0.8732747804265998, "eval_acc_and_f1": 0.8488113032567781, "learning_rate": 0.0, "loss": 0.0025403000922233332, "step": 1700}  
acc = 0.8243478260869566  
acc_and_f1 = 0.8488113032567781  
f1 = 0.8732747804265998  
  
Exp 2  Train and test on MRPC (1600 Train 400 Dev ) T1, D1, 
{"eval_acc": 0.7575, "eval_f1": 0.8391376451077943, "eval_acc_and_f1": 0.7983188225538971, "learning_rate": 4.253731343283582e-05, "loss": 0.5615764591097832, "step": 100}  
{"eval_acc": 0.76, "eval_f1": 0.84, "eval_acc_and_f1": 0.8, "learning_rate": 3.5074626865671645e-05, "loss": 0.30132466070353986, "step": 200}  
{"eval_acc": 0.76, "eval_f1": 0.8241758241758241, "eval_acc_and_f1": 0.7920879120879121, "learning_rate": 2.7611940298507467e-05, "loss": 0.10422512134769932, "step": 300}  
{"eval_acc": 0.7575, "eval_f1": 0.8252252252252251, "eval_acc_and_f1": 0.7913626126126125, "learning_rate": 2.0149253731343285e-05, "loss": 0.04584677870443556, "step": 400}  
{"eval_acc": 0.7725, "eval_f1": 0.8400702987697716, "eval_acc_and_f1": 0.8062851493848857, "learning_rate": 1.2686567164179105e-05, "loss": 0.007963758026307914, "step": 500}  
{"eval_acc": 0.7625, "eval_f1": 0.8312611012433393, "eval_acc_and_f1": 0.7968805506216696, "learning_rate": 5.2238805970149255e-06, "loss": 0.006342481472383952, "step": 600}█ 
acc = 0.765  
acc_and_f1 = 0.7985714285714286  
f1 = 0.8321428571428572  

Exp 3 Train and test on MRPC (800 Train 200 Dev) T1,, D1,,  
{"eval_acc": 0.78, "eval_f1": 0.8345864661654135, "eval_acc_and_f1": 0.8072932330827067, "learning_rate": 3.529411764705883e-05, "loss": 0.4471517898887396, "step": 100}  
{"eval_acc": 0.785, "eval_f1": 0.8542372881355933, "eval_acc_and_f1": 0.8196186440677966, "learning_rate": 2.058823529411765e-05, "loss": 0.0384037708962569, "step": 200}  
{"eval_acc": 0.775, "eval_f1": 0.8464163822525598, "eval_acc_and_f1": 0.8107081911262799, "learning_rate": 5.882352941176471e-06, "loss": 0.0015858456502610351, "step": 300}  
acc = 0.785  
acc_and_f1 = 0.8196186440677966  
f1 = 0.8542372881355933  

EXP 4 Train and Test on all our data(random sample) (1600 Train 400 Dev) T2 D2  
{"eval_acc": 0.7375, "eval_f1": 0.759725400457666, "eval_acc_and_f1": 0.748612700228833, "learning_rate": 4.253731343283582e-05, "loss": 0.6623514720797539, "step": 100}  
{"eval_acc": 0.8275, "eval_f1": 0.8435374149659864, "eval_acc_and_f1": 0.8355187074829933, "learning_rate": 3.5074626865671645e-05, "loss": 0.38900250390172003, "step": 200}  
{"eval_acc": 0.8325, "eval_f1": 0.8466819221967963, "eval_acc_and_f1": 0.8395909610983981, "learning_rate": 2.7611940298507467e-05, "loss": 0.2344598250836134, "step": 300}  
{"eval_acc": 0.8225, "eval_f1": 0.8305489260143198, "eval_acc_and_f1": 0.82652446300716, "learning_rate": 2.0149253731343285e-05, "loss": 0.12879988429136574, "step": 400}  
{"eval_acc": 0.8075, "eval_f1": 0.8144578313253011, "eval_acc_and_f1": 0.8109789156626506, "learning_rate": 1.2686567164179105e-05, "loss": 0.05808970392565243, "step": 500}  
{"eval_acc": 0.815, "eval_f1": 0.820388349514563, "eval_acc_and_f1": 0.8176941747572815, "learning_rate": 5.2238805970149255e-06, "loss": 0.03246088988264091, "step": 600}  
acc = 0.815  
acc_and_f1 = 0.8172560975609755  
f1 = 0.8195121951219512█  

EXP 5 Train and Test on our data(random sample) (800 Train 200 Dev) T2, D2,  
{"eval_acc": 0.74, "eval_f1": 0.7699115044247787, "eval_acc_and_f1": 0.7549557522123893, "learning_rate": 3.529411764705883e-05, "loss": 0.5569935445487499, "step": 100}  
{"eval_acc": 0.745, "eval_f1": 0.7512195121951218, "eval_acc_and_f1": 0.7481097560975609, "learning_rate": 2.058823529411765e-05, "loss": 0.12259279514430091, "step": 200}  
{"eval_acc": 0.75, "eval_f1": 0.7572815533980582, "eval_acc_and_f1": 0.7536407766990292, "learning_rate": 5.882352941176471e-06, "loss": 0.03418641551397741, "step": 300}  
acc = 0.73  
acc_and_f1 = 0.73  
f1 = 0.73  

EXP 6 Train and Test on our data(idioms in Dev not in train)(1600 Train 400 Dev) T3 D3  
{"eval_acc": 0.775, "eval_f1": 0.7836538461538461, "eval_acc_and_f1": 0.7793269230769231, "learning_rate": 4.253731343283582e-05, "loss": 0.6502096274495125, "step": 100}  
{"eval_acc": 0.815, "eval_f1": 0.8355555555555555, "eval_acc_and_f1": 0.8252777777777778, "learning_rate": 3.5074626865671645e-05, "loss": 0.35093204326927663, "step": 200}  
{"eval_acc": 0.8025, "eval_f1": 0.8123515439429929, "eval_acc_and_f1": 0.8074257719714965, "learning_rate": 2.7611940298507467e-05, "loss": 0.18587792794220148, "step": 300}  
{"eval_acc": 0.8, "eval_f1": 0.7938144329896908, "eval_acc_and_f1": 0.7969072164948454, "learning_rate": 2.0149253731343285e-05, "loss": 0.101453266965691, "step": 400}  
{"eval_acc": 0.7925, "eval_f1": 0.7909319899244333, "eval_acc_and_f1": 0.7917159949622166, "learning_rate": 1.2686567164179105e-05, "loss": 0.050527576665626836, "step": 500}  
{"eval_acc": 0.7975, "eval_f1": 0.8019559902200488, "eval_acc_and_f1": 0.7997279951100245, "learning_rate": 5.2238805970149255e-06, "loss": 0.022262947671697474, "step": 600}█  
acc = 0.795  
acc_and_f1 = 0.7984661835748792  
f1 = 0.8019323671497584  

EXP 19 Train and Test on our data(idioms in Dev not in train) (800 Train 200 Dev) T3, D3,  
{"eval_acc": 0.71, "eval_f1": 0.6813186813186813, "eval_acc_and_f1": 0.6956593406593407, "learning_rate": 3.529411764705883e-05, "loss": 0.5713078786432743, "step": 100}  
{"eval_acc": 0.795, "eval_f1": 0.8161434977578476, "eval_acc_and_f1": 0.8055717488789238, "learning_rate": 2.058823529411765e-05, "loss": 0.21139548476785422, "step": 200}  
{"eval_acc": 0.74, "eval_f1": 0.7373737373737373, "eval_acc_and_f1": 0.7386868686868686, "learning_rate": 5.882352941176471e-06, "loss": 0.07025171777931974, "step": 300}  
acc = 0.765  
acc_and_f1 = 0.7700598086124402  
f1 = 0.7751196172248804  


## Effect Transfer Learning
EXP 7 Train T1, Test D2  
{"eval_acc": 0.5225, "eval_f1": 0.6811352253756261, "eval_acc_and_f1": 0.601817612687813, "learning_rate": 4.253731343283582e-05, "loss": 0.5615764591097832, "step": 100}  
{"eval_acc": 0.5175, "eval_f1": 0.6799336650082919, "eval_acc_and_f1": 0.5987168325041459, "learning_rate": 3.5074626865671645e-05, "loss": 0.30132466070353986, "step": 200}█  
{"eval_acc": 0.5075, "eval_f1": 0.6549912434325744, "eval_acc_and_f1": 0.5812456217162871, "learning_rate": 2.7611940298507467e-05, "loss": 0.10422512134769932, "step": 300}  
{"eval_acc": 0.515, "eval_f1": 0.6666666666666665, "eval_acc_and_f1": 0.5908333333333333, "learning_rate": 2.0149253731343285e-05, "loss": 0.04584677870443556, "step": 400}█  
{"eval_acc": 0.5225, "eval_f1": 0.6779089376053963, "eval_acc_and_f1": 0.6002044688026982, "learning_rate": 1.2686567164179105e-05, "loss": 0.007963758026307914, "step": 500}  
{"eval_acc": 0.525, "eval_f1": 0.6790540540540541, "eval_acc_and_f1": 0.602027027027027, "learning_rate": 5.2238805970149255e-06, "loss": 0.006342481472383952, "step": 600}  
acc = 0.525  
acc_and_f1 = 0.6014830508474576  
f1 = 0.6779661016949152  

EXP 8 Train  (T1,), test on D3  
{"eval_acc": 0.5075, "eval_f1": 0.6700167504187605, "eval_acc_and_f1": 0.5887583752093802, "learning_rate": 4.253731343283582e-05, "loss": 0.5615764591097832, "step": 100}  
{"eval_acc": 0.5, "eval_f1": 0.6666666666666666, "eval_acc_and_f1": 0.5833333333333333, "learning_rate": 3.5074626865671645e-05, "loss": 0.30132466070353986, "step": 200}  
{"eval_acc": 0.5, "eval_f1": 0.647887323943662, "eval_acc_and_f1": 0.573943661971831, "learning_rate": 2.7611940298507467e-05, "loss": 0.10422512134769932, "step": 300}  
{"eval_acc": 0.505, "eval_f1": 0.6574394463667821, "eval_acc_and_f1": 0.5812197231833911, "learning_rate": 2.0149253731343285e-05, "loss": 0.04584677870443556, "step": 400}
{"eval_acc": 0.5, "eval_f1": 0.6563573883161511, "eval_acc_and_f1": 0.5781786941580755, "learning_rate": 1.2686567164179105e-05, "loss": 0.007963758026307914, "step": 500}  
{"eval_acc": 0.5, "eval_f1": 0.6563573883161511, "eval_acc_and_f1": 0.5781786941580755, "learning_rate": 5.2238805970149255e-06, "loss": 0.006342481472383952, "step": 600}█  
acc = 0.5  
acc_and_f1 = 0.5781786941580755  
f1 = 0.6563573883161511  

EXP 9 Train  (T1,,), test on D2,  
{"eval_acc": 0.4875, "eval_f1": 0.6397188049209138, "eval_acc_and_f1": 0.5636094024604569, "learning_rate": 3.529411764705883e-05, "loss": 0.4471517898887396, "step": 100}   
{"eval_acc": 0.52, "eval_f1": 0.68, "eval_acc_and_f1": 0.6000000000000001, "learning_rate": 2.058823529411765e-05, "loss": 0.0384037708962569, "step": 200}   
{"eval_acc": 0.515, "eval_f1": 0.6755852842809364, "eval_acc_and_f1": 0.5952926421404683, "learning_rate": 5.882352941176471e-06, "loss": 0.0015858456502610351, "step": 300}  
acc = 0.5175   
acc_and_f1 = 0.59764816360601  
f1 = 0.6777963272120201  

EXP 10 Train (T1,,), test on D3,  
{"eval_acc": 0.48, "eval_f1": 0.637630662020906, "eval_acc_and_f1": 0.558815331010453, "learning_rate": 3.529411764705883e-05, "loss": 0.4471517898887396, "step": 100}  
{"eval_acc": 0.4975, "eval_f1": 0.6633165829145728, "eval_acc_and_f1": 0.5804082914572865, "learning_rate": 2.058823529411765e-05, "loss": 0.0384037708962569, "step": 200}  
{"eval_acc": 0.5025, "eval_f1": 0.6644182124789206, "eval_acc_and_f1": 0.5834591062394603, "learning_rate": 5.882352941176471e-06, "loss": 0.0015858456502610351, "step": 300}  
acc = 0.5  
acc_and_f1 = 0.5816498316498318  
f1 = 0.6632996632996634  

EXP 11 Train  (T1,,), test on D4  
{"eval_acc": 0.49, "eval_f1": 0.6433566433566433, "eval_acc_and_f1": 0.5666783216783217, "learning_rate": 3.529411764705883e-05, "loss": 0.4471517898887396, "step": 100}  
{"eval_acc": 0.525, "eval_f1": 0.6843853820598006, "eval_acc_and_f1": 0.6046926910299003, "learning_rate": 2.058823529411765e-05, "loss": 0.0384037708962569, "step": 200}█  
{"eval_acc": 0.53, "eval_f1": 0.6866666666666666, "eval_acc_and_f1": 0.6083333333333334, "learning_rate": 5.882352941176471e-06, "loss": 0.0015858456502610351, "step": 300}█  
acc = 0.53  
acc_and_f1 = 0.6083333333333334  
f1 = 0.6866666666666666  

EXP 12  Train  (T1,,), test on D5  
{"eval_acc": 0.44583333333333336, "eval_f1": 0.5907692307692308, "eval_acc_and_f1": 0.5183012820512821, "learning_rate": 3.529411764705883e-05, "loss": 0.4471517898887396, "step": 100}  
{"eval_acc": 0.44583333333333336, "eval_f1": 0.6122448979591837, "eval_acc_and_f1": 0.5290391156462585, "learning_rate": 2.058823529411765e-05, "loss": 0.0384037708962569, "step": 200}█  
{"eval_acc": 0.45416666666666666, "eval_f1": 0.6112759643916914, "eval_acc_and_f1": 0.532721315529179, "learning_rate": 5.882352941176471e-06, "loss": 0.0015858456502610351, "step": 300}█  
acc = 0.44583333333333336  
acc_and_f1 = 0.5267514749262536  
f1 = 0.607669616519174  

EXP 13 Train T2,  test on d1,,  
{"eval_acc": 0.665, "eval_f1": 0.7963525835866262, "eval_acc_and_f1": 0.7306762917933132, "learning_rate": 3.529411764705883e-05, "loss": 0.5569935445487499, "step": 100}  
{"eval_acc": 0.58, "eval_f1": 0.7181208053691276, "eval_acc_and_f1": 0.6490604026845638, "learning_rate": 2.058823529411765e-05, "loss": 0.12259279514430091, "step": 200}█  
{"eval_acc": 0.565, "eval_f1": 0.7070707070707071, "eval_acc_and_f1": 0.6360353535353536, "learning_rate": 5.882352941176471e-06, "loss": 0.03418641551397741, "step": 300}  
acc = 0.55  
acc_and_f1 = 0.6176573426573426  
f1 = 0.6853146853146853  
  
  
EXP 14 Train T3, test on d1,,  
{"eval_acc": 0.49, "eval_f1": 0.5920000000000001, "eval_acc_and_f1": 0.541, "learning_rate": 3.529411764705883e-05, "loss": 0.5713078786432743, "step": 100}  
{"eval_acc": 0.45, "eval_f1": 0.5528455284552846, "eval_acc_and_f1": 0.5014227642276423, "learning_rate": 2.058823529411765e-05, "loss": 0.21139548476785422, "step": 200}  
{"eval_acc": 0.41, "eval_f1": 0.40404040404040403, "eval_acc_and_f1": 0.407020202020202, "learning_rate": 5.882352941176471e-06, "loss": 0.07025171777931974, "step": 300}█  
acc = 0.485  
acc_and_f1 = 0.5233510638297871  
f1 = 0.5617021276595744  


#Effects data creation type   
EXP 15 Train on all idiom daniel(T4) test on d5  
{"eval_acc": 0.65, "eval_f1": 0.7062937062937062, "eval_acc_and_f1": 0.6781468531468531, "learning_rate": 3.529411764705883e-05, "loss": 0.6901599037647247, "step": 100}  
{"eval_acc": 0.7625, "eval_f1": 0.7443946188340808, "eval_acc_and_f1": 0.7534473094170404, "learning_rate": 2.058823529411765e-05, "loss": 0.3059185621701181, "step": 200}█  
{"eval_acc": 0.775, "eval_f1": 0.7476635514018691, "eval_acc_and_f1": 0.7613317757009346, "learning_rate": 5.882352941176471e-06, "loss": 0.03216740990174003, "step": 300}█  
acc = 0.7958333333333333  
acc_and_f1 = 0.7784044715447154  
f1 = 0.7609756097560975  

EXP 16 Train on T4 test on D1,,   
{"eval_acc": 0.54, "eval_f1": 0.6912751677852349, "eval_acc_and_f1": 0.6156375838926175, "learning_rate": 3.529411764705883e-05, "loss": 0.6901599037647247, "step": 100}  
{"eval_acc": 0.47, "eval_f1": 0.5954198473282443, "eval_acc_and_f1": 0.5327099236641222, "learning_rate": 2.058823529411765e-05, "loss": 0.3059185621701181, "step": 200}  
{"eval_acc": 0.435, "eval_f1": 0.5637065637065637, "eval_acc_and_f1": 0.49935328185328187, "learning_rate": 5.882352941176471e-06, "loss": 0.03216740990174003, "step": 300}  
acc = 0.425  
acc_and_f1 = 0.48522727272727273  
f1 = 0.5454545454545455  

EXP 17 Train on T5 test on D4  
{"eval_acc": 0.615, "eval_f1": 0.7220216606498195, "eval_acc_and_f1": 0.6685108303249098, "learning_rate": 3.7500000000000003e-05, "loss": 0.4610620604362339, "step": 100}  
{"eval_acc": 0.625, "eval_f1": 0.7292418772563176, "eval_acc_and_f1": 0.6771209386281588, "learning_rate": 2.5e-05, "loss": 0.05544662811793387, "step": 200}  
{"eval_acc": 0.64, "eval_f1": 0.7333333333333333, "eval_acc_and_f1": 0.6866666666666666, "learning_rate": 1.25e-05, "loss": 0.01501974424405489, "step": 300}  
acc = 0.625  
acc_and_f1 = 0.6761363636363635  
f1 = 0.7272727272727272  

EXP 18 Train on T5 test on D1,,  
{"eval_acc": 0.65, "eval_f1": 0.7839506172839505, "eval_acc_and_f1": 0.7169753086419752, "learning_rate": 3.7500000000000003e-05, "loss": 0.4610620604362339, "step": 100}  
"eval_acc": 0.65, "eval_f1": 0.7852760736196319, "eval_acc_and_f1": 0.717638036809816, "learning_rate": 2.5e-05, "loss": 0.05544662811793387, "step": 200}  
"eval_acc": 0.64, "eval_f1": 0.7777777777777777, "eval_acc_and_f1": 0.7088888888888889, "learning_rate": 1.25e-05, "loss": 0.01501974424405489, "step": 300}█  
acc = 0.645  
acc_and_f1 = 0.7139373088685015  
f1 = 0.7828746177370031  

EXP 20 Train on T5 test on D5  
{"eval_acc": 0.925, "eval_f1": 0.923076923076923, "eval_acc_and_f1": 0.9240384615384616, "learning_rate": 3.7500000000000003e-05, "loss": 0.4610620604362339, "step": 100}  
{"eval_acc": 0.9583333333333334, "eval_f1": 0.9565217391304347, "eval_acc_and_f1": 0.957427536231884, "learning_rate": 2.5e-05, "loss": 0.05544662811793387, "step": 200}  
{"eval_acc": 0.9666666666666667, "eval_f1": 0.9649122807017544, "eval_acc_and_f1": 0.9657894736842105, "learning_rate": 1.25e-05, "loss": 0.01501974424405489, "step": 300}  
acc = 0.9583333333333334  
acc_and_f1 = 0.9576149425287357  
f1 = 0.9568965517241379  

EXP 21 Train on T4 Test on T4  
{"eval_acc": 0.735, "eval_f1": 0.7953667953667954, "eval_acc_and_f1": 0.7651833976833977, "learning_rate": 3.529411764705883e-05, "loss": 0.6901599037647247, "step": 100}  
{"eval_acc": 0.995, "eval_f1": 0.9952606635071091, "eval_acc_and_f1": 0.9951303317535545, "learning_rate": 2.058823529411765e-05, "loss": 0.3059185621701181, "step": 200}  
{"eval_acc": 0.995, "eval_f1": 0.9953051643192489, "eval_acc_and_f1": 0.9951525821596244, "learning_rate": 5.882352941176471e-06, "loss": 0.03216740990174003, "step": 300}  
acc = 1.0  
acc_and_f1 = 1.0  
f1 = 1.0  
