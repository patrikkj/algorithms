class Transactions():
    TIE_BREAK_POLICIES = ('insertion', 'alphabetical', 'alphabetical-inv')


    def __init__(self, transactions, break_ties='alphabetical-inv'):
        if isinstance(transactions, list):
            self._transactions = transactions
            self.dict = Transactions._list_to_dict(transactions)
        else:
            self.dict = transactions.copy()

        # Set tie break policy for generating frequent itemsets
        self.break_ties = break_ties

    
    def find_frequent_items(self, threshold, reverse_sort=True):
        """Used in FP-Growth, 1st iteration."""
        # Count the frequency of every item
        items = {}
        for transaction, count in self.dict.items():
            for item in transaction:
                items[item] = items.get(item, 0) + count

        # Sort using the requested tie break policy
        if reverse_sort:
            if self.break_ties == 'insertion':
                sort_func = lambda e: e[1]
            elif self.break_ties == 'alphabetical':
                sort_func = lambda e: (e[1], e[0])
            elif self.break_ties == 'alphabetical-inv':
                sort_func = lambda e: (e[1], -ord(e[0]))
            items = {k: v for k, v in sorted(items.items(), key=sort_func, reverse=True)}
        
        # Filter infrequent items
        frequent_items = {k: v for k, v in items.items() if v >= threshold}
        return frequent_items, items
        
    def filter_infrequent(self, frequent_items, ordering=None):
        # Make sure frequent item ordering is consistent
        if ordering is None:
            ordering = tuple(frequent_items.keys())

        # Rebuild filtered dictionary
        filtered = {}
        for transaction, count in self.dict.items():
            t_filtered = (i for i in transaction if i in frequent_items)
            t_sorted = tuple(sorted(t_filtered, key=ordering.index))
            filtered[t_sorted] = filtered.get(t_sorted, 0) + count
        return Transactions(filtered)

    def to_frozenset(self):
        return {frozenset(k): v for k, v in self}

    @staticmethod
    def _list_to_dict(transactions):
        # Convert list to dictionary representation
        transactions_dict = {}
        for t in transactions:
            # Make sure keys are hashable
            key = t if isinstance(t, tuple) else tuple(t)
            transactions_dict[key] = transactions_dict.get(key, 0) + 1
        return transactions_dict

    @staticmethod
    def _from_string(string, break_ties='alphabetical-inv'):
        lines = string.split('\n')
        tokens = [l.split(',') for l in lines]
        transactions = [''.join([str(list(filter(str.isalpha, s))[0]) for s in t]) for t in tokens]
        return Transactions(transactions, break_ties=break_ties)
        
    
    def __iter__(self):
        return iter(self.dict.items())
        
        
def main():
    T = ['ACFGH', 'BCDEG', 'BCEFH', 'ABCG', 'CDEH', 'ABCGH', 'ABCDGH', 'BCEG', 'ABCFGH', 'ABCDEGH']
    transactions = Transactions(T, break_ties='alphabetical-inv')
    frequent_items = transactions.find_frequent_items(5)
    filtered_transactions = transactions.filter_infrequent(frequent_items)
    print(frequent_items)
    for transaction, count in filtered_transactions:
        print(elem)

if __name__ == '__main__':
    main()
