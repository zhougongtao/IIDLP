import os
import time
import multiprocessing
"""
    generate dataset in different seed 
"""
seeds = [
    2190830449, 3538903541, 1967909051, 1156027684, 2702614907,
    2131527277, 7466345345, 2131516612, 7456363156, 9576586795,
    7252349820, 6284327691, 5624752851, 6384519612, 1571681920,
    2131511616, 1516172371, 4634763472, 1515162735, 6516324354,
] 
def manual_strategy_seed():
    base =[
        "python ManualStrategy-RP.py --data dataA  --noise 0.1 --lr 0.001 --dataset",
        "python ManualStrategy-RP.py --data dataB  --noise 0.1 --lr 0.001 --dataset",
        "python ManualStrategy-RP.py --data dataC  --noise 0.1 --lr 0.001 --dataset",
        "python ManualStrategy-RP.py --data dataD  --noise 0.1 --lr 0.001 --dataset",
        "python ManualStrategy-RP.py --data dataE  --noise 0.1 --lr 0.001 --dataset",
    ]
    configs = []
    for s in base:
        for strategy_num in range(7):
            s2 = s + f" --strategy {strategy_num}"
            for seed in seeds:
                s3 = s2 + f" --seed {seed}"
                configs.append(s3)  
    return configs

def I_ID_LP_RP_seed():
    base =[
        "python I-ID-LP-RP.py --data dataA --noise 0.1 --dataset",
        "python I-ID-LP-RP.py --data dataB --noise 0.1 --dataset",
        "python I-ID-LP-RP.py --data dataC --noise 0.1 --dataset",
        "python I-ID-LP-RP.py --data dataD --noise 0.1 --dataset",
        "python I-ID-LP-RP.py --data dataE --noise 0.1 --dataset",
    ]
    configs = []
    for s in base:
        for seed in seeds:
            s3 = s + f" --seed {seed}"
            configs.append(s3)  
    return configs

"""
    run NN test
"""
def NN_test():
    base =[
        # "python NN_test.py --strategy 0",   # "Random"
        # "python NN_test.py --strategy 1",   # "MaxMean"
        # "python NN_test.py --strategy 2",   # "MaxVar"
        # "python NN_test.py --strategy 3",   # "MaxMean+Var"
        # "python NN_test.py --strategy 4",   # "MaxMean-Var"
        # "python NN_test.py --strategy 5",   # "PreEntropy"
        # "python NN_test.py --strategy 6",   # "MinMaxPro"
        "python NN_test.py --strategy 11",  # IIDLP
    ]
    return base


def run_script(config):
    print(config)
    os.system(config)

if __name__ == "__main__":
    configs = manual_strategy_seed()
    # configs = I_ID_LP_RP_seed()
    # configs = NN_test()
    print(f"{time.asctime()}  start!!!")
    print(len(configs))
    with multiprocessing.Pool(processes=10) as pool:
        pool.map(run_script, configs)
    print(len(configs))