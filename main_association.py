import numpy as np

from association.association import *
from association.transactions import Transactions

np.set_printoptions(linewidth=200)


# transactions = Transactions._from_string("""b,e,g
# b,d,i
# b,d,e,f
# a,d,e
# d,e
# b,d,j
# b,c,d,e,f
# b,d,e,f
# b,e,h""")

# transactions = Transactions._from_string("""T1 f, a, c, d, g, i, m, p
# T2 a, b, c, f, l, m, o
# T3 b, f, h, j, o
# T4 b, c, k, s, p
# T5 a, f, c, e, l, p, m, n""")
    
#transactions = Transactions(['ACD', 'BCE', 'ABCE', 'BE'])
#transactions = Transactions(['ACDK', 'ADK', 'CBDJK', 'CEF', 'BDEJK', 'ADK', 'ABDEJK', 'BDFJK'])
# T = Transactions(["ABCDFGH","DKM","FK","ACGH","ACDDGH","BM","DFKM","ABCDGH",])
T = Transactions(["AB","ACDF","BCDE","ABCD","ABCE"])


MINSUP = 3
MINCONF = 0
# transactions = Transactions(T2, break_ties='alphabetical-inv')


# Initial steps
sort_to_filtered, I_sorted, I_filtered = sort_filter_transactions(T, MINSUP)

# Candidate generation
F, C = apriori(T, MINSUP, method='fk1_fk1')

# Rule generation
filter_func = lambda itemset: sorted(itemset) == sorted('BDE')
rules = apriori_rule(F, MINCONF, filter_func=filter_func)
# rules = apriori_rule(F, MINCONF)
R = structure_rules(rules)

# FP Growth
F2, summary = fp_growth(T, MINSUP)

# Printing
print_transactions(T)
print('\n'*4)
print_dict(I_sorted, title_keys='Item (Sorted)', title_values='Frequency')
print('\n'*4)
print_dict(I_filtered, title_keys='Item (Filtered)', title_values='Frequency')
print('\n'*4)
print_dict(sort_to_filtered, 
           title_keys='Transactions (Sorted)', 
           title_values='Transactions (Filtered)')
print('\n'*4)
print_candidates(C)
print('\n'*4)
print_frequent(F)
print('\n'*4)
print_rules(R)
print('\n'*4)
print_fpgrowth(summary)
