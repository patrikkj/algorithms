
#################################################################
#                        DISCLAIMER                             #
#    I know that implicitly declaring functions by mutating     #
#       global namespace is considered very bad practice.       #
# The purpose of doing it this way was to see if I was able to  #
#  use higher-order decorators to write working psuedocode. :)  #
#################################################################


def _encapsulate(kwargs, default_attrs, namespace=None):
    g = globals() if namespace is None else namespace

    # Create default attribute mapping from set of attributes
    attr_map = {f"{attr}_attr": attr for attr in default_attrs}

    # Update missing arguments from default_kwargs
    for k, v in attr_map.items():
        if k in kwargs:
            attr_map[k] = kwargs[k]

    # Create getters and setters required by nonlocal namespace
    for attr_name, attr_value in attr_map.items():
        _f_name = attr_name.partition('_')[0]
 
        # Create getter
        def get_wrapper(attr_value):
            def get_func(obj):
                return getattr(obj, attr_value)
            return get_func
        _get_name = f"get_{_f_name}"
        g[_get_name] = get_wrapper(attr_value)

        # Create setter
        def set_wrapper(attr_value):
            def set_func(obj, value):
                return setattr(obj, attr_value, value)
            return set_func
        _set_name = f"set_{_f_name}"
        g[_set_name] = set_wrapper(attr_value)

def encapsulate(*attrs, namespace=None):
    """Creates a global getter and setter for every attribute passed to this decorator
        get_attrname(obj)           ->  obj.attrname
        set_attrname(obj, value)    ->  obj.attrname = value
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            _encapsulate(kwargs, attrs, namespace)
            return func(*args, **kwargs)
        return wrapper
    return decorator

class Node:
    def __init__(self, ip=None):
        self.ip = ip
        self.neighs = []
        self.risk = None
        self.pi = None
        self.prob = None

    def __str__(self):
        return f"Ip: {str(self.ip):10}\t \
            Neighbours: {str([(v.ip, w) for v, w in self.neighs]):35}\t \
            Risk: {str(self.risk):10}\t \
            Pi: {str(self.pi.ip if self.pi else self.pi):10}\t \
            Probability: {str(self.prob):10}"


@encapsulate('pi', namespace=g)
def create_path(node, **kwargs):
    # Base case
    if get_pi(node) is None:
        return [node]
    return create_path(get_pi(node)) + [node]


@encapsulate('color', 'd', 'pi', 'neighbours', namespace=g)
def find_shortest_path(graph, start, end, **kwargs):
    # BFS returns end node, with predecessors defining
    # the shortest path to start node if such path exists.
    end_node = bfs(graph, start, end, **kwargs)

    if end_node:
        return create_path(end_node, **kwargs)
