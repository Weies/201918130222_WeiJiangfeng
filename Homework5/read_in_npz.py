import numpy as np
from numpy.core.fromnumeric import sort
a = np.load("char-rnn-snapshot.npz",allow_pickle=True)
Wxh = a["Wxh"] 
Whh = a["Whh"]
Why = a["Why"]
bh = a["bh"]
by = a["by"]
mWxh, mWhh, mWhy = a["mWxh"], a["mWhh"], a["mWhy"]
mbh, mby = a["mbh"], a["mby"]
chars, data_size, vocab_size, char_to_ix, ix_to_char = a["chars"].tolist(), a["data_size"].tolist(), a["vocab_size"].tolist(), a["char_to_ix"].tolist(), a["ix_to_char"].tolist()

print(char_to_ix[':'])

x = np.zeros((vocab_size, 1))
x[char_to_ix[':']] = 1
print(x)



w1=np.dot(Wxh,x)
# for ele in w1:
#     print(ele)

w2 = Wxh[:,9]

w3 = Why[0,:]
w4 = Why[2,:]
print("the didden of teh :")
print(w3)
print(w4)






