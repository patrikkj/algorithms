from .priority_queue import PriorityQueue


class MaxHeap(PriorityQueue):
	def __init__(self, data=None, key_attr=None):
		super().__init__(data=data, compare_keys=lambda k1, k2: k1 > k2, key_attr=key_attr)

	def insert(self, node):
		return super().insert(node)

	def maximum(self):
		return super().top()

	def extract_max(self):
		return super().extract()

	def increase_key(self, i, key):
		return super().modify_key(i, key)