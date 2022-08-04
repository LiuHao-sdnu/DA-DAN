from itertools import count
from turtle import pen
import numpy as np
import  pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader,Dataset
from train import args
import random
class TVdatasets_A(Dataset):
    def __init__(self,Elist,Vlist,train,args,flag,min_len):
        print('Loading training set data......')
        self.Elist = Elist
        self.Vlist = Vlist
        self.train = train
        self.maxlen = args.maxlen
        self.flag = flag
        self.min_len = min_len
        User_a,User_b,User_target = self.getdict(self.Elist,self.Vlist,self.train)
        print('Load complete......')
        self.finally_user_train_a,self.finally_user_target_a = self.sample_seq(User_a,User_target,self.maxlen)
        print(len(self.finally_user_train_a))
        print(len(self.finally_user_target_a))
        # print(self.train_seq)
    def __getitem__(self, index):
        return self.finally_user_train_a[index],self.finally_user_target_a[index]
    def __len__(self):
        return len(list(self.finally_user_train_a.keys()))
    def getdict(self,E_list,V_list,train):
        User_all = defaultdict(list)
        User_target = defaultdict(list)
        with open(train, 'r') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                # Index starts from 1 in order to remove users
                for item in line[1:]:         
                    User_all[line[0]].append(item)
        for k in list(User_all.keys()):
            target=[]
            # Find the label at the end of the mixing sequence
            for index in reversed(range(len(User_all[k])-1)):       
                if User_all[k][index][0] == self.flag:
                    target.append(User_all[k][index])
                if len(target) == 1:
                    del User_all[k][index:]
                    User_target[k] = target[::-1]
                    break
            if len(target) == 0:
                del User_all[k]
                continue
            E=0
            V=0
            for item in list(User_all[k]):
                if item[0] == self.flag:
                    E+=1
                else:
                    V+=1
       #If the sequence length is less than the set minimum length, it will be discarded
            if E <= self.min_len or V <= self.min_len:             
                del User_all[k]
                del User_target[k]
       # Process the item character number and integer number in the data set into a dictionary
        def id_dict(fname):                                       
            itemdict = {}
            with open(fname,'r') as f:
                items =  f.readlines()                 
            for item in items:
                item = item.strip().split('\t')
                itemdict[item[1]] = int(item[0])
            return itemdict
        itemE = id_dict(E_list)
        itemV = id_dict(V_list)
        User_a= defaultdict(list)
        User_b= defaultdict(list)
        for k,v in User_all.items():
            for item in User_all[k]:
                if item[0] == self.flag:
                    User_a[k].append(itemE[item])
                else:
                    User_b[k].append(itemV[item])
            item_list=User_target[k]
            temp=[]
            for i in item_list:
                temp.append(itemE[i])
            User_target[k]=np.array(temp)
        return User_a,User_b,User_target
    def sample_seq(self,user_train_a,user_target,maxlen):
        new_user_id=0
        finally_user_train_a={}
        finally_user_target_a={}
        for user in user_train_a.keys():
            seq = np.zeros([maxlen], dtype=np.int32)
            idx=maxlen-1
            for i in reversed(user_train_a[user]):
                seq[idx]=i
                idx-=1
                if idx==-1:
                    break
            finally_user_train_a[new_user_id] = seq
            finally_user_target_a[new_user_id] = user_target[user]
            new_user_id+=1 
        # Return the sequence interaction information and corresponding label of domain a
        return finally_user_train_a,finally_user_target_a          
class TVdatasets_B(Dataset):
    def __init__(self,Elist,Vlist,train,args,flag,min_len):
        print('Loading training set data......')
        self.Elist = Elist
        self.Vlist = Vlist
        self.train = train
        self.flag=flag
        self.maxlen=args.maxlen
        self.min_len=min_len
        User_a,User_b,User_target=self.getdict(self.Elist,self.Vlist,self.train)
        print('Load complete......')
        self.finally_user_train_b,self.finally_user_target_b=self.sample_seq(User_b,User_target,self.maxlen)
        print(len(self.finally_user_train_b))
        print(len(self.finally_user_target_b))
        # print(self.train_seq)
    def __getitem__(self, index):
        return self.finally_user_train_b[index],self.finally_user_target_b[index]
    def __len__(self):
        return len(list(self.finally_user_train_b.keys()))
    def getdict(self,E_list,V_list,train):
        User_all= defaultdict(list)
        User_target= defaultdict(list)
        with open(train, 'r') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                for item in line[1:]:
                    User_all[line[0]].append(item)
        for k in list(User_all.keys()):
            target=[]
            for index in reversed(range(len(User_all[k])-1)):
                if User_all[k][index][0]==self.flag:
                    target.append(User_all[k][index])
                if len(target) == 1:
                    del User_all[k][index:]
                    User_target[k]=target[::-1]
                    break
            if len(target) == 0:
                del User_all[k]
                continue
            E=0
            V=0
            for item in list(User_all[k]):
                if item[0] == self.flag:
                    E+=1
                else:
                    V+=1
            if E <= self.min_len or V <= self.min_len:
                del User_all[k]
                del User_target[k]
        def id_dict(fname):
            itemdict = {}
            with open(fname,'r') as f:
                items = f.readlines()
            for item in items:
                item = item.strip().split('\t')
                itemdict[item[1]] = int(item[0])
            return itemdict
        itemE = id_dict(E_list)
        itemV = id_dict(V_list)
        User_a = defaultdict(list)
        User_b = defaultdict(list)
        for k,v in User_all.items():
            for item in User_all[k]:
                if item[0] == self.flag:
                    User_a[k].append(itemE[item])
                else:
                    User_b[k].append(itemV[item])
            item_list=User_target[k]
            temp = []
            for i in item_list:
                temp.append(itemE[i])
            User_target[k] = np.array(temp)
        return User_a,User_b,User_target
    def sample_seq(self,user_train_b,user_target_b,maxlen):
        new_user_id = 0
        finally_user_train_b = {}
        finally_user_target_b = {}
        for user in user_train_b.keys():
            seq = np.zeros([maxlen], dtype=np.int32)
            idx=maxlen-1
            for i in reversed(user_train_b[user]):
                seq[idx]=i
                idx -= 1
                if idx == -1:
                    break
            finally_user_train_b[new_user_id] = seq
            finally_user_target_b[new_user_id] = user_target_b[user]
            new_user_id += 1 
        #The label of the B domain returned here is not used, and only the label of the source domain is used during training
        return finally_user_train_b,finally_user_target_b      

class TVdatasets_val(Dataset):
    def __init__(self,Elist,Vlist,train,args,flag,min_len):
        print('Loading training set data......')
        self.Elist = Elist
        self.Vlist = Vlist
        self.train = train
        self.maxlen = args.maxlen
        self.flag = flag
        self.min_len = min_len
        User_b,User_target=self.getdict(self.Elist,self.Vlist,self.train)
        print('Load complete......')
        self.finally_user_train_b,self.finally_user_target_b = self.sample_seq(User_b,User_target,self.maxlen)   
        # print(self.train_seq)
        # print(self.finally_user_train_a)
        # print(self.finally_user_train_b)
        # print(self.finally_user_target_a)
    def __getitem__(self, index):
        return self.finally_user_train_b[index],self.finally_user_target_b[index]
    def __len__(self):
        return len(list(self.finally_user_train_b.keys()))
    def getdict(self,E_list,V_list,train):
        User_all = defaultdict(list)
        User_target = defaultdict(list)
        with open(train, 'r') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                for item in line[1:]:
                    User_all[line[0]].append(item) 
        for k in list(User_all.keys()):
            target=[]
            for index in reversed(range(len(User_all[k]))):                
                if User_all[k][index][0]==self.flag:
                    target.append(User_all[k][index])
        #Take the future three source domain commodities that occur in the target domain sequence as labels, which echo with the paper            
                if len(target) == 3:
                    del User_all[k][index:]                    
                    User_target[k] = target[::-1] 
                    break
          
        def id_dict(fname):
            itemdict = {}
            with open(fname,'r') as f:
                items =  f.readlines()
            for item in items:
                item = item.strip().split('\t')
                itemdict[item[1]] = int(item[0])
            return itemdict
        itemE = id_dict(E_list)
        itemV = id_dict(V_list)
        User_b = defaultdict(list)
        for k,v in User_all.items():
            for item in User_all[k]:
                if item[0] != self.flag:
                    User_b[k].append(itemV[item])
            item_list = User_target[k]
            temp=[]
            for i in item_list:
                temp.append(itemE[i])
            # if len(temp)<3:
            #     while True:
            #         a=random.randint(1,100000)
            #         temp.append(a)
            #         if len(temp)==3:
            #             break
            User_target[k] = np.array(temp)

        print(len(User_b))
        print(len(User_target))
        
        return User_b,User_target
    def sample_seq(self,user_train_b,user_target,maxlen):
        new_user_id=0
        finally_user_target_b = {}
        finally_user_train_b = {}
        new_user_id=0
        for user in user_train_b.keys():
            seq = np.zeros([maxlen], dtype=np.int32)
            idx = maxlen-1
            for i in reversed(user_train_b[user]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break
            finally_user_train_b[new_user_id] = seq
            finally_user_target_b[new_user_id] = user_target[user]
            new_user_id += 1 
        return finally_user_train_b,finally_user_target_b                                          



if __name__=='__main__':

    # datasets_train_a=TVdatasets_A('finalcontruth_info/Elist.txt','finalcontruth_info/Vlist.txt','finalcontruth_info/traindata_sess.txt',args)
    # datasets_train_b=TVdatasets_B('finalcontruth_info/Elist.txt','finalcontruth_info/Vlist.txt','finalcontruth_info/traindata_sess.txt',args)
    # # datasets_train_val=TVdatasets_val('finalcontruth_info/Elist.txt','finalcontruth_info/Vlist.txt','finalcontruth_info/validdata_sess.txt',args)

    # # len_=len(datasets_train_val)
    # data_loader_train_A=DataLoader(datasets_train_a,batch_size=2,shuffle=True)
    # data_loader_train_B=DataLoader(datasets_train_b,batch_size=2,shuffle=True)
    # # data_loader_val=DataLoader(datasets_train_val,batch_size=3,shuffle=True)
    # data_loader_train_A=iter(data_loader_train_A)
    # data_loader_train_B=iter(data_loader_train_B)
    # # data_loader_val=iter(data_loader_val)
    # print(next(data_loader_train_A))
    # print(next(data_loader_train_B))
    # # print(next(data_loader_val))

    data_val=TVdatasets_val('Amazon/movie_id.txt','Amazon/book_id.txt','Amazon/test.txt',args,'M',1)
    data_loader_val=DataLoader(data_val,batch_size=2,shuffle=True)
    data_loader_val=iter(data_loader_val)
    a,b=next(data_loader_val)
    # print(a)