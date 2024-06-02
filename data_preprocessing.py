import re

# with open('data/CNewSum_v2/diverse/train.out', 'r', encoding='utf-8') as fin,  open('data/CNewSum_v2/diverse/train_new.out', 'w', encoding='utf-8') as fin1:
#     for line in fin:
#         ll = line.strip()
#         ll = line.replace(" ", "")
#         fin1.write(ll)

maxl = 0
maxs = ''
with open('data/CNewSum_v2/diverse/test.out', 'r', encoding='utf-8') as fin:
    for line in fin:
        ll = line.strip()
        #ll = line.replace(" ", "")
        tmp = len(ll)
        #if tmp> maxl:
         #   maxs = ll
          #  maxl = tmp
        print(tmp)
        print(ll)
        #fin.flush()
