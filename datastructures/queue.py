class Queue():
    """
    Array implementation of FIFO-queue.
    TODO: Alternative implement using linked list.
    """
    def __init__(self, data=None):
        self.data = data[:] if data else []

    def enqueue(self, x):
        self.data.insert(0, x)

    def dequeue(self):
        if len(self.data) == 0:
            raise IndexError("Stack is empty!")
        return self.data.pop()

    def extend(self, iterable):
        for x in iterable:
           self.enqueue(x)

    def is_empty(self):
        return len(self.data) == 0


    # Dunder methods
    def __len__(self):
        return len(self.data)

    def __str__(self):
        return str([str(elem) for elem in self.data])
