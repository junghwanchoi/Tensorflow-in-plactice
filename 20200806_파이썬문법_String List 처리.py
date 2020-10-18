char = list('hello')
print( char )
# ['h', 'e', 'l', 'l', 'o']

# string => list
words = "python은 프로그래밍을 배우기에 아주 좋은 언어입니다."
words_list = words.split()
print( words_list )
# ['python은', '프로그래밍을', '배우기에', '아주', '좋은', '언어입니다.']

time_str = "10:34:17"
time_str.split(':')
#['10', '34', '17']

# list => string
time_list = ['10', '34', '17']
':'.join(time_list)
#'10:34:17'



