import numpy as np
import python.rustfrc as r

rng = np.random.default_rng()
shp = (3000, 3000)
x = np.full(shp, 50+29j)
r_x = rng.poisson(lam=20, size=shp)
r_x3 = r.pois_gen(shp, 20)
r_x2 = r_x*0.5j
x += r_x*0.987 + r_x2


def square_abs(fourier):
    return np.power(np.abs(fourier), 2)


print(square_abs(x))
print(r.sqr_abs(x.astype(np.complex64)))
