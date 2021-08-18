#  Analyse results of MUSHRA test
#  Evelyn Williams

import numpy as np
import json
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd

#  MUSHRA test settings
results_file = "qualtrics.xlsx" # should be exported from Qualtrics as 'xlsx' file format

# the following variables should be hardcoded as there is no way to detect test questions:
nscr = 80 # number of questions / screens (excluding any training/practice screens)
first_q_name = "Q1_1" # name of first test question in Qualtrics results file (exclude any practice questions)
last_q_name = "Q80_5" # name of last test question


#  Analysis settings
labelstrs = ["1book","2books","4books","vocoded", "SR"] # system names, ordered as in results file (should be the same order as specified in your Qualtreats config file)
reorder = [3,0,1,2,4] # indices to reorder your labelstrs in your plots --> order you want to present your results in analysis / visualisation
pval           = 0.01 # t test p value (if none are significant at p=0.01, you can try p=0.05)

# Visualisation settings
dpi_val = 120 # pixels per inch (pyplot works in real size (inches) & dpi)


def load_qualtrics_results():
    global data_stream
    data_stream = pd.read_excel(results_file).drop([0,0]).reset_index()
    avgtimes = list(data_stream["Duration (in seconds)"])
    # this selects only the test questions from the data (by question name)
    data_stream = data_stream.iloc[:,(data_stream.columns.get_loc(first_q_name)):(data_stream.columns.get_loc(last_q_name))+1]
    # get number of conditions
    ncond = list(data_stream.columns.to_series().str.contains('Q1_')).count(True)
    # get number of subjects
    nsubj = (len(data_stream.index))
    return (ncond, nsubj, avgtimes)

ncond,  nsubj, avgtimes = load_qualtrics_results()
print(np.median(avgtimes))
print(scipy.stats.describe(avgtimes))
def make_array():
    resultArray = np.zeros((nscr, ncond, nsubj)) # make zeros tensor
    for scr in range(nscr): # for each screen (question) in results
        for cond in range(ncond): # for each condition (system/slider) in squestion
            for subj in range(nsubj): # for every subject
                rating = data_stream["Q{0}_{1}".format(scr+1,cond+1)].iloc[subj] # get rating
                if np.isnan(rating):
                    continue # ignore NaN values
                else:
                    resultArray[scr, cond, subj] = rating
    resultArray = resultArray.transpose(0,2,1)
    # reshape so each row is a screen for a given participant, each column is a speech condition
    resultArray = (np.reshape(resultArray, (nscr*nsubj,ncond), order ='F'))
    # remove rows with all Zero ratings
    resultArray = resultArray[~np.all(resultArray == 0, axis=1)]
    # remove rows with all 100 ratings
    resultArray = resultArray[~np.all(resultArray == 100, axis=1)]
    # ▼ UNCOMMENT THIS LINE if you want to save to file for easier inspection ▼
    # pd.DataFrame(resultArray).to_csv("resultArray.txt")
    return resultArray

corrresults = make_array()
print("SHAPE OF RESULTS: {}".format(np.shape(corrresults)))
#  Reorder systems
corrresults = (corrresults[:,reorder])
corrresults = corrresults
np.savetxt('qualtrics-results.txt', corrresults.astype(int), fmt='%i', delimiter=',')

#  Min-max normalization to fit 0-100 range
norm_results = np.zeros(np.shape(corrresults))
for s in range(np.size(corrresults[:,0])):
    tmp = corrresults[s]
    tmp = 100*( tmp - min(tmp)) / ( max(tmp) - min(tmp) )
    norm_results[s] = tmp

# Significance test
dmat = np.zeros((ncond, ncond))
pmat = (np.identity(ncond))/2

# T-test
for c1 in range(ncond):
    for c2 in range(c1+1, ncond):
        ds = norm_results[:, c1] - norm_results[:, c2]
        dmat[c1, c2] = np.mean(ds)
        # t-test assumes that  ratings for a given system are normally distributed
        # if they're not normally distrubuted, use scipy.stats.wilcoxonranktest
        (_, pmat[c1,c2]) = scipy.stats.ttest_ind(norm_results[:,c1], norm_results[:,c2], axis=0, equal_var=True, nan_policy='propagate')


dmat  = dmat - dmat.conj().transpose()
pmat  = pmat + pmat.conj().transpose()
ncmp  = ncond*(ncond-1)/2

# Bonferroni correction!
h0rej = (pmat < pval/ncmp)

# Plotting results
bpfontsize = 18   # font size
labelstrs = [labelstrs[i] for i in reorder]

#  Significance results
xwidth = 1000 # in pixels (you can change this if you want)
ywidth = 1000 # in pixels (you can change this if you want)
fig = plt.figure(figsize=(xwidth/dpi_val, ywidth/dpi_val))
fig.subplots_adjust(bottom=0.2)
h = fig.add_subplot(111)
plt.imshow(np.invert(h0rej),   aspect='equal', cmap=plt.cm.gray, interpolation='nearest') #extent=[0, 1, 0, 1],
plt.title('White pixel (not significant) / black (significant)',fontsize=bpfontsize)
h.set_aspect('equal')
plt.xticks(ticks=np.arange(0, len(labelstrs), 1), labels=labelstrs, fontsize= 9)
plt.yticks(ticks=np.arange(0, len(labelstrs), 1), labels=labelstrs, fontsize=9)
bottom, top = h.get_ylim()
h.set_ylim(bottom+0.5, top-0.5)
plt.savefig('qualtrics-significance.pdf', dpi=dpi_val) # to save pdf version
plt.savefig('qualtrics-significance.png', dpi=dpi_val)
plt.show(h)

# Boxplot
xwidth = 650 #in pixels (you can change this if you want)
ywidth = 500 # in pixels (you can change this if you want)
fig1, ax1 = plt.subplots(figsize=(xwidth/dpi_val, ywidth/dpi_val))
ax1.set_title('MUSHRA results')
ax1.boxplot(norm_results)
bottom, top = h.get_ylim()
plt.xticks(ticks=np.arange(0, ncond+1),  labels=[' ']+labelstrs, fontsize=8)
plt.yticks(ticks=np.arange(0,101,  10), labels = (list(range(0,101,10))))
plt.savefig('qualtrics-boxplot.pdf', dpi=dpi_val) # to save pdf version
plt.savefig('qualtrics-boxplot.png', dpi=dpi_val)
plt.show(ax1)
