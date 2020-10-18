

#############################################
# 예제1 : *.zip
#############################################
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
train_horse_name = os.listdir( train_horse_dir ) # dir로 파일 이름을 모두 저장
train_human_name = os.listdir( train_human_dir )
print( "train_horse_name[:10]:", train_horse_name[:10])
print( "train_human_name[:10]:", train_human_name[:10])
print( "len(train_horse_name):", len(train_horse_name) )
print( "len(train_human_name):", len(train_human_name) )
print( "type(train_horse_name[0]):", type(train_horse_name[0]) )




#############################################
# 예제2 : *.H5
#############################################

import os

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3

SrcUrl = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

local_weights_file = tf.keras.utils.get_file( 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', origin=SrcUrl, extract=False)

pre_trained_model = InceptionV3( input_shape=(150,150,3),
                                 include_top=False,
                                 weights=None )

pre_trained_model.load_weights( local_weights_file )

for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

# 저장하는 법
pre_trained_model.save("MyWeights.h5")

#############################################
# 예제3 : TXT
#############################################

shakespeare_url = "https://homl.info/shakespeare"
filepath = tf.keras.utils.get_file( "shakespeare.txt", shakespeare_url )
with open( filepath ) as f:
    shakespear_text = f.read( )



#############################################
# 예제4 : JSON
#############################################
import json

SrcUrl = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json"
filepath = tf.keras.utils.get_file( "sarcasm.json", SrcUrl )
with open( filepath, 'r' ) as f:
    datastore = json.load( f )

sentences = []
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])



#############################################
# 예제4 : CSV
#############################################

import csv

SrcUrl = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv"
filepath = tf.keras.utils.get_file( "bbc-text.csv", SrcUrl )
# 문서 내용
#
# category,text
# tech,ystems  plasma high-definition tvs  and digital video recorders moving into the living room  the way people watch tv will be radically different in five years  time.  that is according to an expert panel which gathered at the annual consumer electronics show in las vegas to discuss how these new technologies will impact one of our favourite pastimes. with the us leading the trend  programmes and other content will be delivered to viewers via home networks  through cable  satellite  telecoms companies  and broadband service providers to front rooms and portable devices.  one of the most talked-about technologies of ces has been digital and personal video recorders (dvr and pvr). these set-top boxes  like the us s tivo and the uk s sky+ system  allow people to record  store  play  pause and forward wind tv programmes when they want.  essentially  the technology allows for much more personalised tv. they are also being built-in to high-definition tv sets  which are big business in japan and the us  but slower to take off in europe because of the lack of high-definition programming. not only can people forward wind through adverts  they can also forget about abiding by network and channel schedules  putting together their own a-la-carte entertainment. but some us networks and cable and satellite companies are worried about what it means for them in terms of advertising revenues as well as  brand identity  and viewer loyalty to channels. although the us leads in this technology at the moment  it is also a concern that is being raised in europe  particularly with the growing uptake of services like sky+.  what happens here today  we will see in nine months to a years  time in the uk   adam hume  the bbc broadcast s futurologist told the bbc news website. for the likes of the bbc  there are no issues of lost advertising revenue yet. it is a more pressing issue at the moment for commercial uk broadcasters  but brand loyalty is important for everyone.  we will be talking more about content brands rather than network brands   said tim hanlon  from brand communications firm starcom mediavest.  the reality is that with broadband connections  anybody can be the producer of content.  he added:  the challenge now is that it is hard to promote a programme with so much choice.   what this means  said stacey jolna  senior vice president of tv guide tv group  is that the way people find the content they want to watch has to be simplified for tv viewers. it means that networks  in us terms  or channels could take a leaf out of google s book and be the search engine of the future  instead of the scheduler to help people find what they want to watch. this kind of channel model might work for the younger ipod generation which is used to taking control of their gadgets and what they play on them. but it might not suit everyone  the panel recognised. older generations are more comfortable with familiar schedules and channel brands because they know what they are getting. they perhaps do not want so much of the choice put into their hands  mr hanlon suggested.  on the other end  you have the kids just out of diapers who are pushing buttons already - everything is possible and available to them   said mr hanlon.  ultimately  the consumer will tell the market they want.   of the 50 000 new gadgets and technologies being showcased at ces  many of them are about enhancing the tv-watching experience. high-definition tv sets are everywhere and many new models of lcd (liquid crystal display) tvs have been launched with dvr capability built into them  instead of being external boxes. one such example launched at the show is humax s 26-inch lcd tv with an 80-hour tivo dvr and dvd recorder. one of the us s biggest satellite tv companies  directtv  has even launched its own branded dvr at the show with 100-hours of recording capability  instant replay  and a search function. the set can pause and rewind tv for up to 90 hours. and microsoft chief bill gates announced in his pre-show keynote speech a partnership with tivo  called tivotogo  which means people can play recorded programmes on windows pcs and mobile devices. all these reflect the increasing trend of freeing up multimedia so that people can watch what they want  when they want.
# business,worldcom boss  left books alone  former worldcom boss bernie ebbers  who is accused of overseeing an $11bn (£5.8bn) fraud  never made accounting decisions  a witness has told jurors.  david myers made the comments under questioning by defence lawyers who have been arguing that mr ebbers was not responsible for worldcom s problems. the phone company collapsed in 2002 and prosecutors claim that losses were hidden to protect the firm s shares. mr myers has already pleaded guilty to fraud and is assisting prosecutors.  on monday  defence lawyer reid weingarten tried to distance his client from the allegations. during cross examination  he asked mr myers if he ever knew mr ebbers  make an accounting decision  .  not that i am aware of   mr myers replied.  did you ever know mr ebbers to make an accounting entry into worldcom books   mr weingarten pressed.  no   replied the witness. mr myers has admitted that he ordered false accounting entries at the request of former worldcom chief financial officer scott sullivan. defence lawyers have been trying to paint mr sullivan  who has admitted fraud and will testify later in the trial  as the mastermind behind worldcom s accounting house of cards.  mr ebbers  team  meanwhile  are looking to portray him as an affable boss  who by his own admission is more pe graduate than economist. whatever his abilities  mr ebbers transformed worldcom from a relative unknown into a $160bn telecoms giant and investor darling of the late 1990s. worldcom s problems mounted  however  as competition increased and the telecoms boom petered out. when the firm finally collapsed  shareholders lost about $180bn and 20 000 workers lost their jobs. mr ebbers  trial is expected to last two months and if found guilty the former ceo faces a substantial jail sentence. he has firmly declared his innocence.
# sport,tigers wary of farrell  gamble  leicester say they will not be rushed into making a bid for andy farrell should the great britain rugby league captain decide to switch codes.   we and anybody else involved in the process are still some way away from going to the next stage   tigers boss john wells told bbc radio leicester.  at the moment  there are still a lot of unknowns about andy farrell  not least his medical situation.  whoever does take him on is going to take a big  big gamble.  farrell  who has had persistent knee problems  had an operation on his knee five weeks ago and is expected to be out for another three months. leicester and saracens are believed to head the list of rugby union clubs interested in signing farrell if he decides to move to the 15-man game.  if he does move across to union  wells believes he would better off playing in the backs  at least initially.  i m sure he could make the step between league and union by being involved in the centre   said wells.  i think england would prefer him to progress to a position in the back row where they can make use of some of his rugby league skills within the forwards.  the jury is out on whether he can cross that divide.  at this club  the balance will have to be struck between the cost of that gamble and the option of bringing in a ready-made replacement.


sentences = []
labels = []
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
print(len(stopwords))

with open(filepath, 'r', encoding='UTF-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader) # 첫행 건너 뛰기
    for row in reader: # 다음 열
        labels.append(row[0]) # tech business sport
        sentence = row[1]          # tv future in the hands of viewers with home theatre s
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
        sentences.append(sentence) # tv future hands viewers home theatre s...

print(len(labels))
print(len(sentences))
print(sentences[0])


import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


tfds.disable_progress_bar()

###############################
# 예제5 : TFDS
###############################
import tensorflow_datasets as tfds

# TFDS 사용시, Parameter 및 Return값
(raw_train, raw_validation, raw_test), metadata = \
    tfds.load( 'cats_vs_dogs',
               split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
               with_info=True,
               as_supervised=True )

print( raw_train )  # type: PrefetchDataset
print( raw_validation )
print( raw_test )
get_label_name = metadata.features[ 'label' ].int2str

for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow( image )
    plt.title( get_label_name(label) )


###############################
# 예제6 : Pandas, CSV
###############################
import pandas as pd


df = pd.read_csv('sunspots.csv', parse_dates=['Date'], index_col='Date') # 'Data'열은 날짜로 Parsing하고, 'Data'열이 index_col에 해당하는 열
series = df['Monthly Mean Total Sunspot Number'].asfreq('1M')
series.head()

type( series )

series.plot( )

series['1995-01-01':].plot()