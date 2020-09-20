import itertools

import common.utils as utils

from .fp_tree import FPTree
from .transactions import Transactions


########################
# Candidate generation #
########################
def apriori_gen(freq_itemset, f1=None, method='fk1_fk1'):
    if (method := method.lower()) == 'fk1_fk1':
        return _fk1_fk1(freq_itemset)
    elif method == 'fk1_f1':
        return _fk1_f1(freq_itemset, f1)
    return None

def _fk1_fk1(freq_itemset):
    candidates = {}
    buckets = {}
    for itemset in freq_itemset.keys():
        prefix = itemset[:-1]
        buckets.setdefault(prefix, []).append(itemset[-1])
    
    for prefix, bucket in buckets.items():
        for suffix in itertools.combinations(sorted(bucket), 2):
            candidates[prefix + suffix] = 0
    return candidates

def _fk1_f1(freq_itemset, f1):
    candidates = {}
    for prefix in freq_itemset.keys():
        for suffix in f1:
            if suffix[0] not in prefix: 
                candidates[tuple(sorted(set(prefix + suffix)))] = 0
    return candidates


########################
#   Rule generation    #
########################
def apriori_rule(F, minconf, filter_func=None):
    return list(apriori_rulegen(F, minconf, filter_func=filter_func))

def apriori_rulegen(F, minconf, filter_func=None):
    H = {}

    # Produce rules for k=2
    for k, freq_itemsets in list(F.items())[1:]:
        for itemset, support in freq_itemsets.items():
            if filter_func is not None:
                if not filter_func(itemset):
                    continue

            #print(itemset)
            H[1] = {(item, ): 0 for item in itemset}
            for h in list(H[1].keys()):
                left = tuple(sorted(set(itemset) - set(h)))
                if (conf := support / F[k-1][left]) >= minconf:
                    yield ((left, h), conf)

    # Produce remaining rules for k > 2
    for k, freq_itemsets in list(F.items())[1:]:
        for itemset, support in freq_itemsets.items():

            if filter_func is not None:
                if not filter_func(itemset):
                    continue
                
            H[1] = {(item, ): 0 for item in itemset}
            yield from _ap_genrules(itemset, support, F, H, k, m=1, minconf=minconf)

def _ap_genrules(itemset, support, F, H, k, m, minconf):
    if k > m+1:
        H[m+1] = apriori_gen(H[m])
        for h in list(H[m+1].keys()):
            left = tuple(sorted(set(itemset) - set(h)))
            if (conf := support / F[k-m-1][left]) >= minconf:
                yield ((left, h), conf)
            else:
                del H[m+1][h]
        yield from _ap_genrules(itemset, support, F, H, k, m+1, minconf)

def structure_rules(rules):
    R = {}
    for rule in rules:
        (left, right), _ = rule
        #if set('HCI') == set(left + right) or (len(left) + len(right)) != 3:
        #    continue
        R1 = R.setdefault(len(left) + len(right), {}) 
        R1.setdefault(len(right), []).append(rule)
    return {k1: {k2: v2 for k2, v2 in sorted(R1.items())} for k1, R1 in sorted(R.items())}


###############################
# Frequent itemset generation #
###############################
def apriori(transactions, threshold, method='fk1_fk1'):
    # Operate on frozensets for efficent comparison
    transactions = transactions.dict

    k = 1
    C = {1: find_frequency(transactions, candidates=None)}
    F = {1: filter_infrequent(C[1], threshold)}

    while F[k]:
        k += 1
        # Generate candidates
        if method == 'fk1_fk1':
            C[k] = apriori_gen(F[k-1], method='fk1_fk1') 
        else:
            C[k] = apriori_gen(F[k-1], F[1], method='fk1_f1') 

        # Get frequencies of newly generated candidates
        C[k] = find_frequency(transactions, candidates=C[k])

        # Add pruned candidate list
        F[k] = filter_infrequent(C[k], threshold)
    return F, C

def fp_growth(transactions, threshold):
    tree = FPTree(transactions, threshold)
    frequent_itemsets = tree.mine_tree()
    summary = tree.history
    return frequent_itemsets, summary



###############################
#           Misc              #
###############################
def find_frequency(transactions, candidates=None):
    frequencies = {}

    # Find all itemsets of size 1
    if candidates is None:
        for transaction, count in transactions.items():
            for item in transaction:
                frequencies[(item, )] = frequencies.get((item, ), 0) + count
        return frequencies

    # Lookup frequencies of candidates in all transactions
    for c in candidates:
        freq = sum(count for trans, count in transactions.items() if set(c).issubset(set(trans)))
        frequencies[c] = freq
    return frequencies

def filter_infrequent(itemsets, threshold, sort=True):
    itemsets_iter = sorted(itemsets.items()) if sort else itemsets.items()
    return {k: v for k, v in itemsets_iter if v >= threshold}

def sort_filter_transactions(transactions, threshold):
    frequent_items, sorted_items = transactions.find_frequent_items(threshold)
    item_order = list(sorted_items)
    
    # Sorted transactions
    sort_to_filtered = {}
    for t in transactions._transactions:
        t_sorted = ''.join(sorted(t, key=item_order.index, reverse=True))
        t_filtered = ''.join(i for i in t_sorted if i in frequent_items)
        sort_to_filtered[t_sorted] = t_filtered
    return sort_to_filtered, sorted_items, frequent_items


###############################
#          Visualize          #
###############################
def print_dict(dictionary, title_keys='Keys', title_values='Values'):
    c1_header = title_keys
    c2_header = title_values
    data = [[k, v] for k, v in dictionary.items()]
    utils.print_table(data, 
        model={
            c1_header: lambda k: ''.join(k) if isinstance(k, tuple) else str(k),
            c2_header: lambda v: ''.join(v) if isinstance(v, tuple) else str(v),
        }, padding=1)

def print_transactions(T):
    c1_header = 'Transactions'
    data = [[transaction] for transaction in T._transactions]
    utils.print_table(data, 
        model={
            c1_header: lambda transaction: transaction,
        }, padding=1)

def print_transactions_list(T_list, title='Transactions'):
    c1_header = title
    data = [[transaction] for transaction in T_list]
    utils.print_table(data, 
        model={
            c1_header: lambda transaction: transaction,
        }, padding=1)

def print_candidates(C):
    for k, candidates in C.items():
        c1_header = f'Candidate [k={k}]'
        c2_header = 'Support'
        utils.print_table(candidates.items(), 
            model={
                c1_header: lambda candidate: ''.join(candidate),
                c2_header: lambda support: str(support)
            }, padding=1)

def print_frequent(F):
    for k, candidates in F.items():
        c1_header = f'Freq. itemset [k={k}]'
        c2_header = 'Support'
        utils.print_table(candidates.items(), 
            model={
                c1_header: lambda itemset: ''.join(itemset),
                c2_header: lambda support: str(support)
            }, padding=1)

def print_rules(R):
    for k, groups in R.items():
        for m, rules in groups.items():
            c1_header = f'Rule [k={k}, m={m}]'
            c2_header = 'Confidence'
            utils.print_table(rules, 
                model={
                    c1_header: lambda rule: f"{''.join(rule[0])} → {''.join(rule[1])}",
                    c2_header: lambda support: str(round(support, 3))
                }, padding=1)

def print_fpgrowth(summary):
    c1_header = 'Suffix'
    c2_header = 'Conditional Pattern Base'
    c3_header = 'Conditional MFI'
    c4_header = 'Frequent Patterns'
    c3_funcs = {
        dict: lambda freq_items: ('<' + ', '.join(f"{k}:{v}" for k, v in freq_items.items()) + '>') if freq_items else '———',
        list: lambda mfi_tuples: ', '.join(f"<{', '.join(f'{t[0]}:{t[1]}' for t in tup )}>" for tup in mfi_tuples)
    }
    utils.print_table(summary, 
        model={
            c1_header: lambda suffix: ''.join(suffix) if suffix else 'null',
            c2_header: lambda base: ', '.join(f"{''.join(k)}: {v}" for k, v in base.items()) if base else '———',
            c3_header: lambda e: c3_funcs[type(e)](e) if type(e) in c3_funcs else '———',
            c4_header: lambda freq_patterns: ', '.join(f"{''.join(k)}: {v}" for k, v in freq_patterns.items()) if freq_patterns else '———'
        }, 
        sep_func=lambda i, row: row[0] == 'null', 
        padding=2
    )


def main():
    transactions = Transactions._from_string("""110 A, C, F, G, H
111 B, C, D, E, G
112 B, C, E, F, H
113 A, B, C, G
114 C, D, E, H
115 A, B, C, G, H
116 A, B, C, D, G, H
117 B, C, E, G
118 A, B, C, F, G, H
119 A, B, C, D, E, G, H""")
    
    #T = ['ACFGH', 'BCDEG', 'BCEFH', 'ABCG', 'CDEH', 'ABCGH', 'ABCDGH', 'BCEG', 'ABCFGH', 'ABCDEGH']
    # transactions = Transactions(T2, break_ties='alphabetical-inv')
    F, C = apriori(transactions, 5, method='fk1_f1')
    rules = apriori_rule(F, 0)
    R = structure_rules(rules)
    F2, summary = fp_growth(transactions, 5)

    print_candidates(C)
    print_frequent(F)
    print_rules(R)
    print_fpgrowth(summary)

#main()
