
import numpy as np

a = np.arange(10)
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

a = np.array( [1, 2, 3, 10, 20, 30, 0.1, 0.2])

np.where(a < 5, a, 10*a)


# 값, 위치
np.min( a ), np.argmin( a )
# (0.1, 6)

# 값, 위치
np.max( a ), np.argmax( a )
# (30.0, 5)

np.where( a< 1 )
# (array([6, 7]),)

a[ np.where( a< 1 ) ]
# array([0.1, 0.2])

np.where( a>=10, 0, a ) #10보다 크거나 같은 값을 찾아서, 0으로 바꾸고, 아닌것은 그대로 두라는 조건문도 가능합니다.
# array([1. , 2. , 3. , 0. , 0. , 0. , 0.1, 0.2])

