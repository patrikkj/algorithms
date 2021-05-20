import numpy as np
from sklearn.preprocessing import OrdinalEncoder

import classification.decision_tree_logging as decision_tree
import clustering.dbscan as dbscan
from association.association import *
from association.transactions import Transactions
from classification.classification import knn_classifier
from clustering.clustering import k_means, labels_to_clusters
from clustering.hac import *
from common.utils import create_table

np.set_printoptions(linewidth=300)



def run_association():
    T_list = ["ACE",  "BCE",    "BCDE", "CDE",   "DE"]
    T = Transactions(T_list, break_ties="alphabetical-inv")

    MINSUP = 2
    MINCONF = 0.8

    # Initial steps
    sort_to_filtered, I_sorted, I_filtered = sort_filter_transactions(T, MINSUP)

    # Candidate generation
    F, C = apriori(T, MINSUP, method='fk1_fk1')

    # Rule generation
    #filter_func = lambda itemset: sorted(itemset) == sorted('ABH')
    filter_func = lambda itemset: True
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


def run_decision_tree():
    string = """ Nr A B C D Klasse 
    1 L K S 2 J 
    2 H F S 4 J 
    3 H T H 4 J 
    4 L F S 2 N 
    5 L F H 5 N 
    6 H T G 2 N 
    7 L F S 6 N 
    8 L K G 4 N 
    9 H T S 4 J 
    10 L F S 5 N 
    11 L K H 7 N 
    12 H F G 9 J 
    13 L K G 2 N 
    14 L F H 1 J 
    15 L F H 7 N"""

    tokenized_lines, headers, identifiers = create_table(string, 
            remove_chars=',', 
            has_header=True, 
            has_id=True,
            join_items=False,
            remove_id_header=True)
    print(tokenized_lines)

    # Encode data
    encoder = OrdinalEncoder(dtype='int64')
    data_encoded = encoder.fit_transform(tokenized_lines)
    X, y = data_encoded[:, :-1], data_encoded[:, -1]

    # Create model
    attrs = set(range(0, X.shape[1]))
    model, summary = decision_tree.decision_tree(X, y, impurity_func='gini', attrs=attrs)

    # Print summary
    decision_tree.print_summary(summary, headers, encoder.categories_)

    # Create graph from model
    decision_tree.export_tree(model, headers, encoder.categories_, show=False)


def run_dbscan():
    # X = np.array([
    #     [1, 1],
    #     [2, 1],
    #     [2, 4],
    #     [2, 5],
    #     [4, 1],
    #     [5, 1],
    # ])
    X = np.array([
        [2, 4],
        [2, 5],
        [2, 10],
        [2, 11],
        [2, 16],
        [3, 3],
        [3, 10],
        [3, 11],
        [4, 3]
    ])

    # Assignment 3 - 2021
    # X = np.array([
    #     [1, 1], 
    #     [3, 3], 
    #     [3, 4], 
    #     [2, 4], 
    #     [6, 5], 
    #     [7, 6], 
    #     [7, 8], 
    #     [6, 10], 
    #     [12, 4], 
    #     [5, 11], 
    #     [6, 11], 
    #     [5, 10], 
    #     [16, 8], 
    #     [11, 9], 
    #     [13, 8], 
    #     [10, 7], 
    #     [12, 8], 
    #     [15, 3]
    # ])
    min_pts = 4
    eps = 3

    core, border, noise = dbscan.DBSCAN(X, eps, min_pts, norm=1)
    #print(core, border, noise)
    dbscan.plot_clusters(X, core, border, noise)


def run_hac():
    # points = (
    #     Point("P1", 2, 3),
    #     Point("P2", 4, 5),
    #     Point("P3", 6, 4),
    #     Point("P4", 6, 5),
    #     Point("P5", 7, 5),
    #     Point("P6", 7, 12),
    #     Point("P7", 8, 2),
    #     Point("P8", 8, 10)
    # )
    points = (
        Point("A", 4, 3),
        Point("B", 5, 8),
        Point("C", 5, 7),
        Point("D", 9, 2),
        Point("E", 11, 6),
        Point("F", 14, 8)
    )
    
    hac(points, heuristic_func='max', distance_func='euclidean')


def run_k_means():
    X = np.array([
        [4, 8], 
        [4, 10],    
        [4, 13],    
        [5, 3], 
        [5, 7], 
        [7, 11],    
    ])
    init_centroids = np.array([
        [4, 4],
        [5, 8],
        [5, 11],
    ])

    centroids, labels = k_means(X, centroids=init_centroids, k=3, norm=1)
    #print(centroids)
    #print(labels)
    #print(labels_to_clusters(X, labels))


def run_knn():
    X = np.array([
        [4, 8],
        [8, 8],
        [8, 4],
        [6, 7],
        [1, 10],
        [3, 6],
        [2, 4],
        [1, 7],
        [6, 4],
        [6, 2],
        [6, 3],
        [4, 3],
        [4, 4],
    ])
    y = np.array([
        [1],
        [1],
        [1],
        [1],
        [2],
        [2],
        [2],
        [2],
        [3],
        [3],
        [3],
        [3],
        [3],
    ])
    X_pred = np.array([
        [6, 6],
        [4, 6],
        [4, 5],
        [2, 6],
    ])

    # Build model
    model = knn_classifier(X, y, default_k=3, norm=1)

    # Classify new data
    y_hat = model(X_pred)



run_association()
#run_decision_tree()
#run_dbscan()
#run_hac()
#run_k_means()
#run_knn()
