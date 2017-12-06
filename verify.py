fp = open('300wlist.txt')
fw = open('300w_id_list_.txt','w')
map1 = {}
map2 = {}
map3={}
map4={}
i_helen = 1
i_afw = 1
i_ibug = 1
i_lfwp = 1
datas = fp.readlines()
for line in datas:
    a = line.split('/')[-1]
    if line.split('/')[1]=='HELEN':

        name = a.split('_')[1]
        print(name)
        if not name in map1:
            map1[name]=i_helen
            i_helen= i_helen+1
    if line.split('/')[1]=='AFW':
        name = a.split('_')[1]
        print(name)
        if not name in map2:
            map2[name]=i_afw
            i_afw= i_afw+1
    if line.split('/')[1]=='IBUG':
        name = a.split('_')[2]
        print(name)
        if not name in map3:
            map3[name]=i_ibug
            i_ibug= i_ibug+1
    if line.split('/')[1]=='LFPW':
        name = a.split('_')[3]
        print(name)
        if not name in map4:
            map4[name]=i_lfwp
            i_lfwp= i_lfwp+1
            
helennum = len(map1)
afwnum = len(map2)
ibugnum = len(map3)
lfwpnum = len(map4)

for line in datas:
    a = line.split('/')[-1]
    if line.split('/')[1]=='HELEN':
        print(2)
        name = a.split('_')[1]
        num = map1[name]
        im = line.split('\n')[0]
        fw.write(im + ' '+ str(num)+'\n')
    if line.split('/')[1]=='AFW':
        print(3)
        name = a.split('_')[1]
        num = map2[name]+helennum
        im = line.split('\n')[0]
        fw.write(im + ' '+ str(num)+'\n')
    if line.split('/')[1]=='IBUG':
        print(4)
        name = a.split('_')[2]
        num = map3[name]+helennum+afwnum
        im = line.split('\n')[0]
        fw.write(im + ' '+ str(num)+'\n')
    if line.split('/')[1]=='LFPW':
        print(line)
        name = a.split('_')[3]
        num = map4[name]+helennum+afwnum+ibugnum
        im = line.split('\n')[0]
        fw.write(im + ' '+ str(num)+'\n')
fp.close()
fw.close()
print helennum+afwnum+ibugnum+lfwpnum
