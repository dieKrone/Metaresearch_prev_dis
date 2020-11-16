# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:02:06 2020

@author: vankronp
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pickle
import pandas as pd
import matplotlib as mpl

root = r'S:\Fakultaet\MFZ\NWFZ\AGdeHoz\Philipp\Data\Metaresearch'
filename = r'\test_consensus.csv'

df = pd.read_csv(root + filename, sep = ';', encoding=r'ISO-8859-1')


col_reason = np.arange(2,201,3)

vec_reason=[]

for i in col_reason:
    vec_reason = np.append(vec_reason, df[f'cit{i}'])
    
idx1 = np.where(vec_reason==1)[0]
idx2 = np.where(vec_reason==2)[0]
idx3 = np.where(vec_reason==3)[0]
idx4 = np.where(vec_reason==4)[0]
idx6 = np.where(vec_reason==6)[0]
idx7 = np.where(vec_reason==7)[0]
idx8 = np.where(vec_reason==8)[0]
idx9 = np.where(vec_reason==9)[0]
idx10 = np.where(vec_reason==10)[0]
idx13 = np.where(vec_reason==13)[0]

categories = ['method (1)', 'protocol (2)', 'prior design (3)', 'manual (4)', 'info (6)', 'credit (7)', 'software (7.1)', 'source (8)', 'formula (9)', 'other (10)']
colorlist = ['k', 'b', 'lightgreen', 'tomato', 'r', 'orange', 'g', 'm', 'cornflowerblue', 'aquamarine']
explode = (0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0)

x = [len(idx1), len(idx2), len(idx3), len(idx4), len(idx6), len(idx7), len(idx13), len(idx8), len(idx9), len(idx10)]

# plt.pie(x, labels=categories, colors=colorlist, pctdistance=5)
# plt.title('classification of methodological citations')


fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    if pct < 2:
        return ""
    else:
        return "{:.1f}%\n({:d})".format(pct, absolute)

wedges, texts, autotexts = ax.pie(x, colors=colorlist, autopct=lambda pct: func(pct, x),
                                  textprops=dict(color="w"))

ax.legend(wedges, categories,
          title="categories",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.title('classification of methodological citations (test dataset)')





col_p = np.arange(3,201,3)

vec_p=[]

for i in col_p:
    vec_p = np.append(vec_p, df[f'cit{i}'])

idxprob = np.where(vec_p=='probable')[0]
idxposs = np.where(vec_p=='possible')[0]
idxnone = np.where(vec_p=='none')[0]

x2 = [len(idxprob), len(idxposs), len(idxnone)+len(idx6)+len(idx7)+len(idx13)+len(idx8)+len(idx9)+len(idx10)]
labels=['probable', 'possible', 'none']
# plt.pie(x, labels=['probable', 'possible', 'none'], colors=colorlist, pctdistance=5)
# plt.title('shortcut or not a shortcut')


fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    if pct < 2:
        return ""
    else:
        return "{:.1f}%\n({:d})".format(pct, absolute)

wedges, texts, autotexts = ax.pie(x2, colors=colorlist, autopct=lambda pct: func(pct, x2),
                                  textprops=dict(color="w"))

ax.legend(wedges, labels,
          title="categories",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.title('probable, possible or not a shortcut (test dataset)')




#how many citations per paper?
sum(x)/160

#how many papers use the supplements for methodological explanaition
len(np.where(df['is_supp']==1)[0])

#ow many papers use a repository
len(np.where(df['is_rep']==1)[0])

#what are the repositories used
w = np.array(df['rep_name'])
w = w.astype('str')

w = w[~np.isnan(w)]

w.dtype()



##############################################################################
#total number of citations - violin plots
##############################################################################

print('Shape of data{}'.format(df.shape))
df.head

method_cols = np.arange(16, 215, 3)

cit_no = []

for i in range(np.size(df, 0)):
    
    row_temp = np.array(df.iloc[i,method_cols])
    
    row_temp = row_temp.astype('float')
    row_temp = row_temp[~np.isnan(row_temp)]
    
    cit_no = np.append(cit_no, len(row_temp))



            
    
    
dict = {'cit_no': cit_no}

data = pd.DataFrame.from_dict(dict)

sns.set_style("ticks")
my_pal2 = {"cit_no": 'green'}
plt.figure()
ax1 = sns.violinplot(data=data, palette=my_pal2)
plt.title('Biology - number of citations per publication')
plt.ylabel('number of citations')




