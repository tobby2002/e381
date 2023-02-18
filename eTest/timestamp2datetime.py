import datetime
ut = 1676681394062/1000

dt = datetime.datetime.fromtimestamp( ut ).strftime('%Y-%m-%d %H:%M:%S')
print( dt ) # 2014-07-24 21:16:36