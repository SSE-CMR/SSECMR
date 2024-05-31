from scheme import *
from collections import defaultdict
import random
import argparse
import pandas as pd
import time
import json
import pickle
import os
import gc

# Precautions
# When running the script, 'local' is recommended to set to True to save time.
# When s = 1, the solution is equivalent to an unoptimized solution, and the execution time will be very long.
# When testing the time of QueryGen, MAP testing should not be performed.

parser = argparse.ArgumentParser()
parser.add_argument('--db', default='mir', help='Database name')        
parser.add_argument('--r', type=int, default='5', help='Search radius')   
parser.add_argument('--h', type=int, default='32', help='Hashlen')  
parser.add_argument('--s', type=int, default='2', help='Number of subcodes')   
parser.add_argument('--v', type=bool, default=False, help='Whether to verify the correctness of results')
parser.add_argument('--t', type=int, default='2000', help='Number of query')
parser.add_argument('--mode', default='range',help='Search for a range of raddi of a specific radius ') # ['range', 'specific']
args = parser.parse_args()
modal = ('BI', 'BT') # T to I
local = True

databases = ['coco', 'mir', 'nus']
data = ['BI', 'BT', 'L']
vector_lengths = [32, 64, 128]
folder = '.Serialization'

# User arguments
db = args.db
R = args.r
maxR = args.r
hashLen = args.h
ss = args.s
virify = args.v
testnum = args.t
queryList = []
mode = args.mode 

# Default arguments
blocksize = 16
hashLenByte = hashLen >> 3

IMI = [defaultdict(set) for i in range(ss)]
enIMI = [{} for i in range(ss)]
database = []

def singlecalMap(queryidx, queryL, residx, resL, topk):
    mergeres = []
    for subres in residx:
        mergeres.extend(subres)
    decryptedmergeres = [int.from_bytes(decrypt(j, 'd'*16, 16), byteorder='big') for j in mergeres]
    totalgnd = (np.dot(queryL[queryidx, :], resL.transpose()) > 0).astype(np.float32)
    gnd = (np.dot(queryL[queryidx, :], resL[decryptedmergeres].transpose()) > 0).astype(np.float32)  
    tgnd = gnd[:topk]
    tsum = np.sum(tgnd)
    if tsum == 0:
        return 0
    count = np.linspace(1, tsum, int(tsum))
    tindex = np.asarray(np.where(tgnd == 1)) + 1.0  
    topkmap_ = np.mean(count/tindex)
    return topkmap_

def pack_Search(idx, query, R, virify, qu_L, re_L, topk):
    # QueryGen
    start_token_generation = time.time()
    token = user.tokenGen(query, R, dataOwener.Gen())
    end_token_generation = time.time()

    # Search
    start_search = time.time()
    res = user.hammingSearch(enIMI, token, R, partitions_dict)
    end_search = time.time()

    # Human verification
    reslen = sum([len(i) for i in res])

    # Use brute force to verify
    if virify is True:
        viridb = defaultdict(set)
        can = getCandidates(query, hashLen, R)
        for id, i in enumerate(database):
            viridb[i].add(id)
        realres = []
        for i in can:
            realres.extend(viridb[i])
        assert len(realres) == reslen

    if qu_L is not None and re_L is not None:
        topkmap_ = singlecalMap(idx, qu_L, res, re_L, topk)
    else:
        topkmap_  = -1

    res = {
        'Initialization': end_init - start_init,
        'Encryption': end_encryption - start_encryption,
        'Token generation': end_token_generation - start_token_generation,
        'Search': end_search - start_search,
        'Result number': reslen,
        'topkmap': topkmap_,
    }

    return res

start_init = time.time()
if not os.path.exists(folder):
    os.makedirs(folder)

IMI_file = os.path.join(folder, f'IMI_{db}_{hashLen}_{ss}.pkl')
database_file = os.path.join(folder, f'database_{db}_{hashLen}.pkl')

if local and os.path.exists(IMI_file) and os.path.exists(database_file):
    with open(IMI_file, 'rb') as f:
        IMI = pickle.load(f)
    with open(database_file, 'rb') as f:
        database = pickle.load(f)
        n = len(database)
else:
    interval = (hashLenByte // ss)
    if db in databases:
        df = pd.read_csv(f'./Data/{db}/{hashLen}/re_{modal[0]}.csv', header=None)
        df.replace(-1, 0, inplace=True)
        df = df.astype(int)
        n = len(df)
        for i in range(n):
            content = int(''.join([str(i) for i in list(df.iloc[i])]), 2)
            content = content.to_bytes(hashLenByte, byteorder='big')
            database.append(content)
            for j, II in enumerate(IMI):
                l, r = interval * j, interval * (j + 1)
                II[content[l: r]].add(i)
    else:
        vectors = np.load(f'./Data/{db}/{hashLen}/random_vectors.npy', 'r')
        for idx, vector in enumerate(tqdm(vectors, desc="Processing vectors")):
            content = vector.tobytes()
            database.append(content)
            for j, II in enumerate(IMI):
                l, r = interval * j, interval * (j + 1)
                II[content[l: r]].add(idx)
        n = len(database)
        print("Processing vectors done.")
        del vectors
        gc.collect()
    
    with open(IMI_file, 'wb') as f:
        pickle.dump(IMI, f)
    with open(database_file, 'wb') as f:
        pickle.dump(database, f)

if db in databases:
    df = pd.read_csv(f'./Data/{db}/{hashLen}/qu_{modal[1]}.csv', header=None)
    df.replace(-1, 0, inplace=True)
    df = df.astype(int)
    for idx, i in enumerate(range(len(df))):
        if idx == testnum:
            break
        content = int(''.join([str(i) for i in list(df.iloc[i])]), 2)
        content = content.to_bytes(hashLenByte, byteorder='big')
        queryList.append((idx, content))
else:
    for i in range(testnum):    
        randinx = random.randint(0, n - 1)
        queryList.append((randinx, database[randinx]))

if virify is False:
    del database
    gc.collect()
    
maxElemNum = []
print("For hashLen: ", hashLen, ", s: ", ss, ", R: ", R, ", db: ", db, ", mode: ", mode)
for idx, II in enumerate(IMI):
    maxElemNum.append(max([len(item) for item in II.values()]))

m = {i: f'msg{i}' for i in range(n)}
v = [f'' for i in range(n)]
M = (m, v)
maxLenV = max([len(i) for i in v])
dataOwener = Label(lenV=maxLenV, blocksize=blocksize)
user = User(blocksize=blocksize, lenV=maxLenV, hashLen=hashLen, K=dataOwener.Gen(), ss=ss)

partitions_dict = {}

for nn in range(R + 1):  
    partitions = partition_ordered(nn, ss, hashLen // ss)  
    partitions_dict[(nn, ss)] = partitions
end_init = time.time()

# IndexBuld
start_encryption = 0
end_encryption = 0

enIMI_file = os.path.join(folder, f'enIMI_{db}_{hashLen}_{ss}.pkl')
c_file = os.path.join(folder, f'c_{db}_{hashLen}.pkl')

if local and os.path.exists(enIMI_file) and os.path.exists(c_file):
    with open(enIMI_file, 'rb') as f:
        enIMI = pickle.load(f)
    with open(c_file, 'rb') as f:
        c = pickle.load(f)
else:
    start_encryption = time.time()
    for i, II in enumerate(IMI):
        enIMI[i] = dataOwener.Enc(dataOwener.Gen(), II)
    end_encryption = time.time()
    c = dataOwener.EncData(M)
    with open(enIMI_file, 'wb') as f:
        pickle.dump(enIMI, f)
    with open(c_file, 'wb') as f:
        pickle.dump(c, f)

res = {
    'topkmap': 0, 
}

if db in databases:
    qu_L = np.genfromtxt(f'./Data/{db}/{hashLen}/qu_L.csv', delimiter=',', dtype=np.float32)
    re_L = np.genfromtxt(f'./Data/{db}/{hashLen}/re_L.csv', delimiter=',', dtype=np.float32)
else :
    qu_L = None
    re_L = None
topk = 50
if mode == 'specific':
    myRange = range(R, R+1)
else:
    myRange = range(maxR+1)

# This loop can be optimized with binary search
for R in myRange:
    cntlesstopk = 0
    for idx, query in queryList:
        tepres = pack_Search(idx, query, R, virify, qu_L, re_L, topk)
        result_number = tepres['Result number']
        res['topkmap'] += tepres['topkmap']
        if result_number < topk:
            cntlesstopk += 1

    if cntlesstopk > 100: # This value can be a little bigger to get result, set smaller to get acurate result
        res['topkmap'] = 0
        continue

    res['topkmap'] /= testnum
    print("The topkmap is: ", res['topkmap'])
    break

    