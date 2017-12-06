import os

fp = open('300wlist.txt','w')
filepath = '300W_LP'
#for i in range(1,501):
#    num = "%03d"%i
#    filepath = path+num
#i = 0  
for file in os.listdir(filepath):
    if os.path.isdir(filepath+'/'+file):
        if len(file.split('_'))==2:
            continue
        if file == 'landmarks':
            continue
        if file=='Code':
            continue
        for file_ in os.listdir(filepath+'/'+file):
            if file_.split('.')[1] == 'jpg':
                fp.write(filepath+'/'+file+'/'+file_.split('.')[0]+'\n')
#                fp.write(str(i)+'\n')
    
