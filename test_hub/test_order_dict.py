import collections
print ("Regular dictionary")
d={}
d['a']=None
for k,v in d.items():
    print (k,v)
print ("\nOrder dictionary")
d1 = collections.OrderedDict()
d1.setdefault('a')
d1.setdefault('a')
d1.setdefault('b')
d1.setdefault('v')
for k,v in d1.items():
    print (k,v)