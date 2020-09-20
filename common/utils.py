from collections import defaultdict
import itertools


def groupby(iterable, key=None):
    if key is None:
        key = lambda x: x
    groups = defaultdict(list)
    for element in iterable:
        lookup = key(element)
        groups[lookup].append(element)
    return groups

def generate_subsets(iterable):
    for r in range(1, len(iterable) + 1):
        yield from itertools.combinations(iterable, r)


def print_table(data, model, padding=2, sep_func=None):
     # Preprocess data
    data_strings = []
    for record in data:
        data_strings.append(list(func(elem) for elem, func in zip(record, model.values())))

    # Calculate column widths and create dynaimcally sized templates
    headers = model.keys()
    col_sizes = [max(len(e) for e in column) + 2*padding for column in zip(*(data_strings + [headers]))]
    meta_template = '|' + '|'.join(["{{:^{}}}" for _ in range(len(headers))]) + '|'
    template = meta_template.format(*col_sizes)
    header = template.format(*headers)
    sep_line = '—'*len(header)
    dash_line = '–'*len(header)

    print(f'{sep_line}')
    print(header)
    print(sep_line)
    for i, record in enumerate(data_strings):
        if sep_func is not None and sep_func(i, record) and i != 0:
            print(dash_line)
        print(template.format(*record))
    print(sep_line)


def create_table(string, remove_chars=',', has_header=True, has_id=True, join_items=False, remove_id_header=True):
    # Remove unwanted delimiters
    if isinstance(remove_chars, dict):
        for old, new in remove_chars.items():
            string = string.replace(old, new)
    else:
        for c in remove_chars:
            string = string.replace(c, '')

    # Parse lines
    lines = string.split('\n')

    # Remove empty lines
    lines = [line for line in lines if bool(line.strip())]
    
    # Handle header
    if has_header:
        headers, lines = lines[0], lines[1:]
        headers = headers.split()
        if remove_id_header and has_id:
            headers = headers[1:]
    
    # Handle ID's
    tokenized_lines = [line.split() for line in lines]
    if has_id:
        id_tuples = [(line[0], line[1:]) for line in tokenized_lines]
        identifiers, tokenized_lines = list(zip(*id_tuples))

    # Join items
    if join_items:
        tokenized_lines = [''.join(tokens) for tokens in tokenized_lines]

    ret_headers = headers if has_header else None
    ret_identifiers = identifiers if has_id else None
    return tokenized_lines, ret_headers, ret_identifiers
