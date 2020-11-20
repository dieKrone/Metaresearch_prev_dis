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
import seaborn as sns

root = r'S:\Fakultaet\MFZ\NWFZ\AGdeHoz\Philipp\Data\Metaresearch'
filename_bio = r'\BiologyConsensusSheetCleaned.csv'
filename_psych = r'\Psychiatry_Abstraction_List_Consensus.csv'

df_bio = pd.read_csv(root + filename_bio, sep = ';', encoding=r'ISO-8859-1')
df_psych = pd.read_csv(root + filename_psych, sep = ';', encoding=r'ISO-8859-1')
df_neuro = pd.read_pickle(r'S:\Fakultaet\MFZ\NWFZ\AGdeHoz\Philipp\Data\Metaresearch\df_neuro.pkl', compression='infer')

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

fields = ['Neuroscience', 'Biology', 'Psychiatry']

method_cols_bio = np.arange(18, np.size(df_bio, 1), 3)
method_cols_psych = np.arange(19, np.size(df_psych, 1), 3)
method_cols_neuro = np.arange(17, np.size(df_neuro, 1), 3)
p_cols_bio = np.arange(19, np.size(df_bio, 1), 3)
p_cols_psych = np.arange(20, np.size(df_psych, 1), 3)
p_cols_neuro = np.arange(21, np.size(df_neuro, 1), 3)
pro_neuro = []
pos_neuro = []
no_neuro = []
pro_bio = []
pos_bio = []
no_bio = []
pro_psych = []
pos_psych = []
no_psych = []
cit_no_psych = []
cit_no_bio = []
cit_no_neuro = []



for i in range(np.size(df_psych, 0)):
    
    row_temp = np.array(df_psych.iloc[i,method_cols_psych])
    p_temp = np.array(df_psych.iloc[i,p_cols_psych])
    
    row_temp = row_temp.astype('float')
    row_temp = row_temp[~np.isnan(row_temp)]
    p_temp = p_temp.astype('str')
    pro_idx = np.where(p_temp=='probable')[0]
    pos_idx = np.where(p_temp=='possible')[0]
    no_idx = np.where(p_temp=='none')[0]
    
    cit_no_psych = np.append(cit_no_psych, len(row_temp))
    pro_psych = np.append(pro_psych, len(pro_idx))
    pos_psych = np.append(pos_psych, len(pos_idx))
    no_psych = np.append(no_psych, len(no_idx))

for i in range(np.size(df_neuro, 0)):
    
    row_temp = np.array(df_neuro.iloc[i,method_cols_neuro])
    p_temp = np.array(df_neuro.iloc[i,p_cols_neuro])
    
    row_temp = row_temp.astype('float')
    row_temp = row_temp[~np.isnan(row_temp)]
    pro_idx = np.where(p_temp=='probable')[0]
    pos_idx = np.where(p_temp=='possible')[0]
    no_idx = np.where(p_temp=='none')[0]
    
    cit_no_neuro = np.append(cit_no_neuro, len(row_temp))
    pro_neuro = np.append(pro_neuro, len(pro_idx))
    pos_neuro = np.append(pos_neuro, len(pos_idx))
    no_neuro = np.append(no_neuro, len(no_idx))

for i in range(np.size(df_bio, 0)):
    
    row_temp = np.array(df_bio.iloc[i,method_cols_bio])
    p_temp = np.array(df_bio.iloc[i,p_cols_bio])
    
    row_temp = row_temp.astype('float')
    row_temp = row_temp[~np.isnan(row_temp)]
    pro_idx = np.where(p_temp=='probable')[0]
    pos_idx = np.where(p_temp=='possible')[0]
    no_idx = np.where(p_temp=='none')[0]
    
    cit_no_bio = np.append(cit_no_bio, len(row_temp))
    pro_bio = np.append(pro_bio, len(pro_idx))
    pos_bio = np.append(pos_bio, len(pos_idx))
    no_bio = np.append(no_bio, len(no_idx))
    
    
array_psych = np.zeros(len(cit_no_bio)-len(cit_no_psych))
array_psych[:] = np.nan
cit_no_psych = np.concatenate((cit_no_psych, array_psych))
pro_psych = np.concatenate((pro_psych, array_psych))
pos_psych = np.concatenate((pos_psych, array_psych))
no_psych = np.concatenate((no_psych, array_psych))

array_neuro = np.zeros(len(cit_no_bio)-len(cit_no_neuro))
array_neuro[:] = np.nan
cit_no_neuro = np.concatenate((cit_no_neuro, array_neuro))
pro_neuro = np.concatenate((pro_neuro, array_neuro))
pos_neuro = np.concatenate((pos_neuro, array_neuro))
no_neuro = np.concatenate((no_neuro, array_neuro))

dict = {'neuro': cit_no_neuro,
        'bio': cit_no_bio,
        'psych': cit_no_psych}

dict_pro = {'pro_neuro': pro_neuro,
            'pro_bio': pro_bio,
            'pro_psych': pro_psych}

dict_pos = {'pos_neuro': pos_neuro,
            'pos_bio': pos_bio,
            'pos_psych': pos_psych}

dict_no = {'no_neuro': no_neuro,
           'no_bio': no_bio,
           'no_psych': no_psych}

data = pd.DataFrame.from_dict(dict)

d_pro = pd.DataFrame.from_dict(dict_pro)
d_pos = pd.DataFrame.from_dict(dict_pos)
d_no = pd.DataFrame.from_dict(dict_no)

Neuro_7colors = np.array(['#FFFCEB',
                            '#FFF6C2',
                            '#FFED85',
                            '#FFE347',
                            '#E0BF00',
                            '#B89C00',
                            '#8F7900'])

 

Psych_7colors = np.array(['#FFEEEB',
                            '#FFCBC2',
                            '#FF9785',
                            '#FF6347',
                            '#E02200',
                            '#B81C00',
                            '#8F1500'])

 

Bio_7colors = np.array(['#EDF3FD',
                            '#C8DAF9',
                            '#90B5F3',
                            '#598FEE',
                            '#1558CB',
                            '#1248A5',
                            '#0E3881'])

Neuro_4colors =  Neuro_7colors[[0, 2, 4, 6]]
Psych_4colors =  Psych_7colors[[0, 2, 4, 6]]
Bio_4colors =  Bio_7colors[[0, 2, 4, 6]]

Neuro_3colors =  Neuro_7colors[[0, 3, 6]]
Psych_3colors =  Psych_7colors[[0, 3, 6]]
Bio_3colors =  Bio_7colors[[0, 3, 6]]

Neuro_1color =  Neuro_7colors[3]
Psych_1color =  Psych_7colors[3]
Bio_1color =  Bio_7colors[3]


sns.set_style("ticks")
my_pal2 = {"neuro": Neuro_1color, 'bio': Bio_1color, 'psych': Psych_1color}
plt.figure()
ax = sns.violinplot(data=data, palette=my_pal2)

for violin, alpha in zip(ax.collections[::2], [0.5, 0.5, 0.5]):
    violin.set_alpha(alpha)
    
plt.title('Number of citations per publication over fields')
plt.ylabel('number of citations')


#probable, possible, none
s_types = ['pro', 'pos', 'no']
s_type_full = ['probable', 'possible', 'none']
data = [d_pro, d_pos, d_no]

for i, typ in enumerate(s_types):
    sns.set_style("ticks")
    my_pal = {"{}_neuro".format(typ): Neuro_1color, '{}_bio'.format(typ): Bio_1color, '{}_psych'.format(typ): Psych_1color}
    plt.figure()
    ax = sns.violinplot(data=data[i], palette=my_pal)
    
    for violin, alpha in zip(ax.collections[::2], [0.5, 0.5, 0.5]):
        violin.set_alpha(alpha)
        
    plt.title('Number of {}-shortcuts per publication over fields'.format(s_type_full[i]))
    plt.ylabel('number of citations')
    plt.ylim((-2.5,18))

np.nansum(cit_no_neuro)
np.nansum(cit_no_bio)
np.nansum(cit_no_psych)


