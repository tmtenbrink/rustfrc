import rustfrc as r
import diplib as dip
import numpy as np
import time

rng = np.random.default_rng()
img = dip.ImageRead('Benton')(0)[:388, :388]
x = np.array(img)
x = np.repeat(x[:, :, np.newaxis], 1000, axis=2)
# half_x = np.rint(x/2).astype(int)
# print(half_x)
start = time.time_ns()
a = rng.binomial(x, 0.5)
# a = r.binom_split(x)
b = x - a
# for i in range(100):
#     a = rng.binomial(half_x, 0.5)
#     b = half_x - a
end_time = time.time_ns()
print(str(float(end_time - start)/1e9) + " s")

#
