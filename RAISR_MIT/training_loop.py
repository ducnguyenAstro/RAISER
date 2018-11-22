import os
import time

cmd = ["python3 train.py -d 16 -R 8 -fn train/stack_T11SMV_20181001T183149_B04_01 -z","python3 train.py -v -q -d 16 -R 8 -fn train/stack_T11SMV_20181001T183149_B04_02 -z", "python3 train.py -v -q -d 16 -R 8 -fn train/stack_T11SMV_20181001T183149_B04_03 -z", "python3 train.py -v -q -d 16 -R 8 -fn train/stack_T11SMV_20181001T183149_B04_04 -z"]


elapsed = []

for i in range(len(cmd)):
    print("Trainning for set {}  :".format(i+1))
    t0 = time.time()
    try:
        print(cmd[i])
        os.system(cmd[i])
    except OSError:
        print ("ERROR")
    elapsed +=[time.time()-t0]   
    #time.sleep(60*5)

print( "COMPLETED in {} seconds!".format(sum(elapsed)))
for i in range(len(elapsed)):
    print('Time for {} training set: {} hours'.format(i+1,elapsed[i]/3600))




