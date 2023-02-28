import datetime
# ut = 1676681394062/1000
#
# dt = datetime.datetime.fromtimestamp( ut ).strftime('%Y-%m-%d %H:%M:%S')
# print( dt ) # 2014-07-24 21:16:36


def timestamp2datetime(x):
    return str(datetime.datetime.fromtimestamp(float(x) / 1000).strftime('%Y-%m-%d %H:%M:%S'))

t = timestamp2datetime(1676681394062)
print(t)