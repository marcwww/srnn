import random
def gen_line(line):
    with open('./data/aa-bb.txt','a+') as f:
            print(line,file=f)

alphabet=[chr(ord('a')+i) for i in range(26)]
for _ in range(1000):
    ch1=random.sample(alphabet,k=1)[0]
    n=int(random.random()*10)+1
    src=[ch1 for _ in range(n)]
    tar=[ch1 for _ in range(n)]
    line=' '.join(src)+'\t'+' '.join(tar)
    gen_line(line)


