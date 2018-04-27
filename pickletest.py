import sys
from time import sleep

for i in range(1,5):
    print('\r',i, end = '')
    sys.stdout.flush()
    sleep(1)
print('\r')
print(10)