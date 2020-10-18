
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

####################
# 이미지 가져오기
####################
i = misc.ascent() #misc라는 곳에서 계산이미지를 바로 받을 수 있음

# 이미지 그리기
plt.grid( False )
plt.gray()
plt.axis('off')
plt.imshow( i )
plt.show()

#타입, 사이즈 구하기
print( type(i)  ) # [list] or (tuple) or {dictionary} or set 인지 확인
size_x = i.shape[0] # numpy array 인 경우 shape[0]은 가로, shape[1]은 세로
size_y = i.shape[1]
print( size_x, size_y )

#copy
i_transformed = np.copy( i )


####################
#filter 적용예
####################
filter1 = [ [ 0, 1, 0],
            [ 1,-4, 1],
            [ 0, 1, 0] ]
filter2 = [ [-1,-2,-1],
            [ 0, 0, 0],
            [ 1, 2, 1] ]
filter3 = [ [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1] ]

weight = 1
filter = filter2

for x in range(1,size_x-1):
  for y in range(1,size_y-1):
      convolution = 0.0
      convolution = convolution + (i[x - 1, y-1] * filter[0][0])
      convolution = convolution + (i[x, y-1] * filter[0][1])
      convolution = convolution + (i[x + 1, y-1] * filter[0][2])
      convolution = convolution + (i[x-1, y] * filter[1][0])
      convolution = convolution + (i[x, y] * filter[1][1])
      convolution = convolution + (i[x+1, y] * filter[1][2])
      convolution = convolution + (i[x-1, y+1] * filter[2][0])
      convolution = convolution + (i[x, y+1] * filter[2][1])
      convolution = convolution + (i[x+1, y+1] * filter[2][2])
      convolution = convolution * weight
      if(convolution<0):
        convolution=0
      if(convolution>255):
        convolution=255
      i_transformed[x, y] = convolution


# 이미지 그리기
plt.grid( False )
plt.gray()
plt.axis('off')
plt.imshow( i_transformed )
plt.show()

####################
# POOLING 적용예
####################
new_x = int(size_x/2)
new_y = int(size_y/2)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
  for y in range(0, size_y, 2):
    pixels = []
    pixels.append(i_transformed[x, y])
    pixels.append(i_transformed[x+1, y])
    pixels.append(i_transformed[x, y+1])
    pixels.append(i_transformed[x+1, y+1])
    newImage[int(x/2),int(y/2)] = max(pixels)

# Plot the image. Note the size of the axes -- now 256 pixels instead of 512
plt.gray()
plt.grid(False)
plt.imshow(newImage)
#plt.axis('off')
plt.show()



