class Stack():
    """
    Array implementation of LIFO-queue.
    TODO: Alternative implement using linked list.
    """
    def __init__(self, data=None):
        self.data = data[:] if data else []
        self.top = len(self.data) - 1

    def push(self, x):
        self.data.append(x)
        self.top += 1

    def pop(self):
        if self.top == -1:
            raise IndexError("Stack is empty!")
        self.top -= 1
        return self.data.pop()

    def extend(self, iterable):
        for x in iterable:
            push(x)

    def is_empty(self):
        return len(self.data) == 0


    # Dunder methods
    def __len__(self):
        return len(self.data)

    def __str__(self):
        return str([str(elem) for elem in self.data])

