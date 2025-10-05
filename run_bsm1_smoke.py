'''
from tqdm import tqdm
from bsm2_python import BSM1OL  # or BSM1AC for aeration control

m = BSM1OL()                 # build the plant
n = 2000                     # short run ~ a few thousand steps
for i in tqdm(range(n), desc="BSM1 short run"):
    m.step(i)

print("OK: ran", n, "steps")
'''

# run_bsm1_full.py
from bsm2_python import BSM1OL
m = BSM1OL()

m.simulate()
