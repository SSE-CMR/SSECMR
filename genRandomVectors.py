import numpy as np
import os
from tqdm import tqdm

def generate_random_vectors(num_lines, vector_length, filename):
    vectors = np.empty((num_lines, vector_length//8), dtype=np.uint8)
    
    for i in tqdm(range(num_lines)):
        vectors[i] = np.packbits(np.random.randint(0, 2, vector_length, dtype=np.uint8))
    
    np.save(filename, vectors)

n = 200 * (10**4)
D = [f'diy{i*2}M' for i in range(1, 6)] 
H = [64]

for i in range(1, 6):
    for hashlen in H:
        db = D[i-1]
        print(f"Generating ./Data/{db}/{hashlen}...")
        os.makedirs(f'./Data/{db}/{hashlen}', exist_ok=True)
        generate_random_vectors(n*i, hashlen, f'./Data/{db}/{hashlen}/random_vectors.npy')
