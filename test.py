import python.rustfrc as r
import numpy as np
import time

init_x = (np.ones((900, 700, 100))*24.31)
init_y = (np.ones((900, 700, 100))*30).astype(np.int32)
init_z = np.ones((900, 700, 50))*30


# start = time.perf_counter()
# x = r.binom_split(init_x)
# end = time.perf_counter()
# print(str(end - start) + " s")
#
#
# start2 = time.perf_counter()
# rng = np.random.default_rng()
# y = rng.binomial(init_y, 0.5)
# end2 = time.perf_counter()
# print(str(end2 - start2) + " s")