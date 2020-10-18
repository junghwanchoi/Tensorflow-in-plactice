import os
import zipfile

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

SrcUrl = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"


# 방법#1
#저장위치: C:\Users\jungh\.keras\datasets\horse-or-human.zip
#path_to_zip = tf.keras.utils.get_file( 'horse-or-human.zip', origin=SrcUrl, extract=True)
#PATH        = os.path.join( os.path.dirname(path_to_zip), 'horse-or-human')

# 방법#2
local_zip = tf.keras.utils.get_file( 'horse-or-human.zip', origin=SrcUrl, extract=False) #저장위치: C:\Users\jungh\.keras\datasets\horse-or-human.zip
zip_ref     = zipfile.ZipFile( local_zip, 'r')
zip_ref.extractall('/tmp/horse-or-human') # D:\tmp\horse-or-human
# zip_ref.extractall('./tmp/horse-or-human') # *.py 파일 path에 생성됨
zip_ref.close()

# Directory path
train_horse_dir = os.path.join('/tmp/horse-or-human/horses') #경로를 병합하여 새 경로 생성
train_human_dir = os.path.join('/tmp/horse-or-human/humans') #경로를 병합하여 새 경로 생성
# file name
train_horse_name = os.listdir( train_horse_dir )
train_human_name = os.listdir( train_human_dir )
print( "train_horse_name[:10]:", train_horse_name[:10])
print( "train_human_name[:10]:", train_human_name[:10])
print( "len(train_horse_name):", len(train_horse_name) )
print( "len(train_human_name):", len(train_human_name) )
print( "type(train_horse_name[0]):", type(train_horse_name[0]) )



# 4 * 4 이미지 표시
nrows = 4
ncols = 4

pic_index = 0
fig = plt.gcf() # get the current figure
fig.set_size_inches( ncols*4, nrows*4 )

pic_index += 8
next_horse_pix = [ os.path.join(train_horse_dir, fname) for fname in train_horse_name[pic_index-8:pic_index] ]
next_human_pix = [ os.path.join(train_human_dir, fname) for fname in train_human_name[pic_index-8:pic_index] ]

for i, img_path in enumerate( next_horse_pix+next_human_pix ): # i = 0, 1, ... (16-1)
    # Set up subplot
    sp = plt.subplot( nrows, ncols, i+1 ) # sp: subplot 1,2,...,16
    sp.axis( 'off' )

    img = mpimg.imread( img_path )
    plt.imshow( img )

plt.show( )


