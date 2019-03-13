import numba as nb
from collections import deque
import time


a=deque([1,2,3])
a.append(4)
print(a)
a.appendleft(6)
print(a)
