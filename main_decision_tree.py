import numpy as np
from sklearn.preprocessing import OrdinalEncoder

import classification.decision_tree_logging as decision_tree

np.set_printoptions(linewidth=200)



# headers = "Nr, A, B, C, D, E, Class".split(', ')
# table = """1, L, K, R, J, 2, J
# 2, H, F, S, N, 4, J
# 3, H, T, S, N, 4, J
# 4, L, F, S, J, 2, N
# 5, L, F, G, N, 5, N
# 6, H, T, G, N, 2, N
# 7, L, F, S, N, 6, N
# 8, L, K, G, N, 4, N
# 9, H, T, H, N, 2, J
# 10, L, F, S, J, 5, N
# 11, L, K, B, N, 7, N
# 12, H, F, B, N, 9, J
# 13, L, K, R, J, 2, N
# 14, L, F, H, J, 1, N
# 15, L, F, H, N, 7, N"""

# headers = "Nr A B C D Class".split(' ')
# table = """1 L F R 2 J
# 2 H T S 4 J
# 3 H T S 4 J
# 4 L F S 2 N
# 5 H F G 5 N
# 6 H T G 2 N
# 7 L F S 6 N
# 8 H K G 4 N
# 9 H T H 2 J
# 10 H F S 5 N
# 11 H K B 7 N
# 12 L F B 9 N
# 13 L K R 2 N
# 14 L F H 1 N
# 15 L F H 7 N"""

# headers = """Kundenr Alder Biltype Kjørelengde Bonus Skade""".split(' ')
headers = """Dag Turnering Sted Tidspunkt Resultat""".split(' ')
headers = """ID Alder Inntekt Arbeidstype Kredittverdighet Bilkjøper""".split(' ')
table = """1 <=30 Høy Fulltid Passe Nei
2 <=30 Høy Fulltid Høy Nei
3 31-40 Høy Fulltid Passe Ja
4 >40 Middels-høy Fulltid Passe Ja
5 >40 Lav Deltid Passe Ja
6 >40 Lav Deltid Høy Nei
7 31-40 Lav Deltid Høy Ja
8 <=30 Middels-høy Fulltid Passe Nei
9 <=30 Lav Deltid Passe Ja
10 >40 Middels-høy Deltid Passe Ja
11 <=30 Middels-høy Deltid Høy Ja
12 31-40 Middels-høy Fulltid Høy Ja
13 31-40 Høy Deltid Passe Ja
14 >40 Middels-høy Fulltid Høy Nei"""

data_raw = [line.split(' ') for line in table.split('\n')]

# Encode data
encoder = OrdinalEncoder(dtype='int64')
data_encoded = encoder.fit_transform(data_raw)
X, y = data_encoded[:, :-1], data_encoded[:, -1]

# Create model
attrs = set(range(1, X.shape[1]))
model, summary = decision_tree.decision_tree(X, y, impurity_func='entropy', attrs=attrs)

# Print summary
decision_tree.print_summary(summary, headers, encoder.categories_)

# Create graph from model
decision_tree.export_tree(model, headers, encoder.categories_, show=False)


#print(X)
#print(y)
#print(encoder.categories_)
#print(encoder.inverse_transform(np.hstack([X, (y.reshape(-1, 1))])))
