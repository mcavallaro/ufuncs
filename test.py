from nmf import *
import numpy as np

import UNpPoissonBetaUtils as BP

a = nmf([1,11,3,4], [1.1])
print a

a = nmf([2.1], [1,2,3,4])
print a