

#################################################
# Pandas DataFrame, Drop
#################################################
    df = pd.DataFrame(np.arange(12).reshape(3, 4),
                      columns=['A', 'B', 'C', 'D'])
    df
       A  B   C   D
    0  0  1   2   3
    1  4  5   6   7
    2  8  9  10  11


    df.drop(['B', 'C'], axis=1)
       A   D
    0  0   3
    1  4   7
    2  8  11

    df.drop(columns=['B', 'C'])
       A   D
    0  0   3
    1  4   7
    2  8  11

    df.drop([0, 1])
       A  B   C   D
    2  8  9  10  11

#################################################
#1. csv 파일 불러오기: read_csv()
#################################################
'''
            ID LAST_NAME AGE
            --+---------+----
            1   KIM       30
            2   CHOI      25
            3   LEE       41
            4   PARK      19
            5   LIM       36
'''

#       csv 파일은 구분자(separator, delimiter)를 명시적으로 ',' (comma)라고 지정해주지 않아도 알아서 잘 불러옵니다.


        import pandas as pd
        csv_test = pd.read_csv('D:/Work/PyCharm/pythonProject1/test_csv_file.csv')

#       DataFrame.shape 을 사용해서 행(row)과 열(column)의 개수를 확인
        csv_test.shape # number of rows, columns
        csv_test


#################################################
# 2. 구분자 '|' 인 text 파일 불러오기 : sep='|'
#################################################
#    만약 구분자가 탭(tab) 이라면 sep = '\t' 을 입력해줍니다.
        text_test = pd.read_csv('D:/Work/PyCharm/pythonProject1/test_text_file.txt', sep='|')
        text_test

#################################################
#  3. 파일 불러올 때 index 지정해주기 : index_col
#################################################
#     만약에 위의 예에서 첫번째 열인 'ID'라는 이름의 변수를 Index 로 지정해주고 싶으면 index_col=0 (위치)이나 index_col='ID' 처럼 직접 변수 이름을 지정해주면 됩니다.
#

#       pass the column number you wish to use as the index:
        text_test = pd.read_csv('D:/Work/PyCharm/pythonProject1/test_text_file.txt', sep='|', index_col=0)
        text_test

#       pass the column name you wish to use as the index:
        text_test = pd.read_csv('D:/Work/PyCharm/pythonProject1/test_text_file.txt', sep='|', index_col='ID')

#  4. 변수 이름(column name, header) 이 없는 파일 불러올 때 이름 부여하기
#      : names=['X1', 'X2', ... ], header=None
#
#
        text_test = pd.read_csv('D:/Work/PyCharm/pythonProject1/test_text_file.txt', sep='|', names=['ID', 'A', 'B', 'C', 'D'], header=None, index_col='ID')
        text_test

#################################################
#  5. 유니코드 디코드 에러, UnicodeDecodeError: 'utf-8' codec can't decode byte
#################################################
#
#       "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc1 in position 26: invalid start byte"
#       이럴 경우에는 Windows에서 많이 사용하는 'CP949'로 아래처럼 encoding을 설정해서 text, csv 파일 불러오기를 해보시기 바랍니다.
        f = pd.read_csv('directory/file', sep='|'', encoding='CP949')

        #    혹시 encoding='CP949' 로 해도 안되면 encoding='latin' ('ISO-8859-1' 의 alias) 도 한번 시도해보시기 바랍니다.
        f = pd.read_csv('directory/file', sep='|'', encoding='latin')


#################################################
#  6. 특정 줄은 제외하고 불러오기: skiprows = [x, x]
#################################################
#     skip rows 옵션을 사용하여 첫번째와 두번째 줄은 제외하고 csv 파일을 DataFrame으로 불러와보겠습니다.
#
#       skip 1st and 2nd rows (do not read 1, 2 rows)
        csv_2 = pd.read_csv("D:/Work/PyCharm/pythonProject1/test_csv_file.csv", skiprows = [1, 2])
        csv_2

#################################################
#  7. n 개의 행만 불러오기: nrows = n
#################################################
#     csv 파일의 위에서 부터 3개의 행(rows) 만 DataFrame으로 불어와보겠습니다.
#
#       read top 3 rows only
        csv_3 = pd.read_csv("D:/Work/PyCharm/pythonProject1/test_csv_file.csv", nrows = 3)
        csv_3

#################################################
# 8. 사용자 정의 결측값 기호 (custom missing value symbols)
#################################################
#
#       어떤 문서에 숫자형 변수에 결측값이 '??'라는 표시로 입력이 되어있다고 한다면, 이를 pandas DataFrame으로
#       불러읽어들였을 경우 float나 int로 인식되어 불러오는 것이 아니라 string으로 인식해서 '??'를 결측값이 아니라
#       문자형으로 불러오게 됩니다. 이럴 경우 '??'를 결측값이라고 인식하라고 알려주는 역할이 na_values = ['??'] 옵션입니다.
        df = pd.read_csv( 'D:/Work/PyCharm/pythonProject1/test_text_file.txt', na_values=['?', '??', 'N/A', 'NA', 'nan', 'NaN', '-nan', '-NaN', 'null'] )
        df

#################################################
# 9. 데이터 유형 설정 (Setting the data type per each column)
#################################################
#    tpye 옵션인 사전형(dictionary)으로 각 칼럼(key)별 데이터 유형(value)를 짝을 지어서 명시적으로 설정해 줄 수 있습니다.
#    날짜/시간 형태(date/time format)의 데이터의 경우 infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates 등의
#    시계열 데이터 형태에 특화된 옵션들이 있습니다. 자세한 내용은 아래의 pandas 매뉴얼을 참고하시기 바랍니다.
#      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

        df = pd.read_csv('C:/Users/Administrator/Documents/Python/test_text_file.txt', dtype = {"ID": int, "LAST_NAME": str, "AGE": float} )

#################################################
#  10. Python pandas의 read_csv() 함수를 사용하여 csv file, text file 을 읽어와 DataFrame을 만들 때 날짜/시간 (Date/Time)이
#      포함되어 있을 경우 이를 날짜/시간 형태(DateTime format)에 맞도록 파싱하여 읽어오는 방법
#################################################
#    (1) 날짜/시간 포맷 지정 없이 pd.read_csv() 로 날짜/시간 데이터 읽어올 경우

        df = pd.read_csv('date_sample', sep=",", names=['date', 'id', 'val']) # no datetime parsing
        df
        '''
        date	id	val
        0	1/5/2020 10:00:00	1	10
        1	1/5/2020 10:10:00	1	12
        2	1/5/2020 10:20:00	1	17
        3	1/5/2020 10:00:00	2	11
        4	1/5/2020 10:10:00	2	14
        5	1/5/2020 10:20:00	2	16
        '''

        df.info()
        '''
        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 6 entries, 0 to 5
        Data columns (total 3 columns):
        date    6 non-null object          <-- not datetime format
        id      6 non-null int64
        val     6 non-null int64
        dtypes: int64(2), object(1)
        memory usage: 272.0+ bytes
        '''

        df_date = pd.read_csv("date_sample",
                              sep=",",
                              names=['date', 'id', 'val'],
                              parse_dates=['date'],
                              dayfirst=True, # May 1st
                              infer_datetime_format=True)


#################################################
# Pandas에서의 Period과 빈도 처리
#################################################
        # 몇일, 몇개월 몇 분기, 몇 해 같은 기간은 Period 클래스로 표현 가능
        p=pd.Period(2007, freq='A-DEC')
        p #2007년 1월1일 부터 같은 해 12월 31일 까지
        # Period('2007', 'A-DEC')
        p+12
        # Period('2019', 'A-DEC')
        rng = pd.period_range('1/1/2019', '6/30/2019', freq='M')
        #일반적인 기간 범위는 period_range 함수를 통해 생성할 수 있다.
        rng
        #PeriodIndex(['2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06'], dtype='period[M]', freq='M')

        '''
        freq 인수로 특정한 날짜만 생성되도록 할 수도 있다. 많이 사용되는 freq 인수값은 다음과 같다.
        
        s: 초
        T: 분
        H: 시간
        D: 일(day)
        B: 주말이 아닌 평일
        W: 주(일요일)
        W-MON: 주(월요일)
        M: 각 달(month)의 마지막 날
        MS: 각 달의 첫날
        BM: 주말이 아닌 평일 중에서 각 달의 마지막 날
        BMS: 주말이 아닌 평일 중에서 각 달의 첫날
        WOM-2THU: 각 달의 두번째 목요일
        Q-JAN: 각 분기의 첫달의 마지막 날
        Q-DEC: 각 분기의 마지막 달의 마지막 날
        보다 자세한 내용은 다음 웹사이트를 참조한다.
        
        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        '''
#################################################
# Period의 빈도 변환
#################################################
        # 10.5.1 Period의 빈도 변환
        #A-DEC :주어진 월(12월)의 마지막 날을 가리키는 연간 주기
        p= pd.Period('2019', freq='A-DEC') #2019.1.1 ~2019.12.31
        p
        # Period('2019', 'A-DEC')
        p.asfreq('M',how='start') #asfreq를 통해 월간 빈도로 변환
        # Period('2019-01', 'M')
        p.asfreq('M', how='end')
        # Period('2019-12', 'M')
        p = pd.Period('2019', freq='A-JUN') #2018.7.1~2019.6.31
        p
        # Period('2019', 'A-JUN')
        p.asfreq('M', 'start')
        # Period('2018-07', 'M')
        p.asfreq('M', 'end')
        # Period('2019-06', 'M')
        p = pd.Period('2019-8','M')
        p.asfreq('A-JUN') # 2019년 8월이 2020년에 속하게 됨.
        # Period('2020', 'A-JUN')
        rng = pd.period_range('2006','2009', freq='A-DEC')
        rng
        # PeriodIndex(['2006', '2007', '2008', '2009'], dtype='period[A-DEC]', freq='A-DEC')
        ts=Series(np.random.randn(len(rng)), index = rng)
        ts
        '''
        2006   -0.323017
        2007    0.424659
        2008   -1.222213
        2009    1.328498
        Freq: A-DEC, dtype: float64
        '''
        ts.asfreq('D',how='start')
        '''
        2006-01-01   -0.323017
        2007-01-01    0.424659
        2008-01-01   -1.222213
        2009-01-01    1.328498
        Freq: D, dtype: float64
        '''