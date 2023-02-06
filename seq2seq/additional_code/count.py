import os
root = 'expert_weights'
lst = os.listdir(root)
cnt = 0
for l in lst:
    lst2 = os.listdir(root+'/'+l)
    cnt+=len(lst2)
print(f'Total number of adapters: {cnt}')