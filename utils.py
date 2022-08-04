import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import pandas as pd
def get_eval(predlist, truelist, klist):#return recall@k and mrr@k
    # predlist=predlist.cpu().numpy()
    # truelist=truelist.cpu().numpy()
    recall = []
    mrr = []
    predlist = predlist.argsort()
    for k in klist:
        recall.append(0)
        mrr.append(0)
        templist = predlist[:,-k:]#the result of argsort is in ascending 
        i = 0
        while i < truelist.shape[0]:
            max_list=[]
            for j in range(truelist.shape[1]):
                pos = torch.argwhere(templist[i]==(truelist[i,j]-1))#pos is a list of positions whose values are all truelist[i]
                if len(pos)>0:
                    max_list.append(pos[0][0])
            if len(max_list) >0:
                recall[-1] += 1
                mrr[-1] += 1/(k-max(max_list))
            else:
                recall[-1] += 0
                mrr[-1] += 0
            i += 1
    return recall, mrr

# def get_eval(predlist, truelist, klist):#return recall@k and mrr@k
#     # predlist=predlist.cpu().numpy()
#     # truelist=truelist.cpu().numpy()
#     recall = []
#     mrr = []
#     predlist = predlist.argsort()
#     for k in klist:
#         recall.append(0)
#         mrr.append(0)
#         templist = predlist[:,-k:]#the result of argsort is in ascending 
#         i = 0
#         while i < truelist.shape[0]:
#             for j in range(truelist.shape[1]):
#                 pos = torch.argwhere(templist[i]==(truelist[i,j]-1))#pos is a list of positions whose values are all truelist[i]
#                 if len(pos)>0:
#                     break
#             if len(pos) >0:
#                 recall[-1] += 1
#                 mrr[-1] += 1/(k-pos[0][0])
#             else:
#                 recall[-1] += 0
#                 mrr[-1] += 0
#             i += 1
#     return recall, mrr