import datastructures.max_heap as max_heap


###############################
# Comparison-based algorithms #
###############################

def insertion_sort(array):
    """Sort using the insertion sort algorithm.

    Features:
        Best case:          O(n) 
        Average case:       O(n²)
        Worst case:         O(n²)
        Space complexity:   O(1)
        In-place:           Yes
        Stable:             Yes
    """
    for i in range(1, len(array)):
        key = array[i]
        j = i - 1

        # Insert arr[j] into the sorted subsequence
        while (j >= 0) and (array[j] > key):
            arr[j + 1] = arr[j]
            j -= 1
        array[j + 1] = key
    return array


def merge_sort(array):
    """Sort using the merge sort algorithm.

    Features:
        Best case:          O(n log n) 
        Average case:       O(n log n)
        Worst case:         O(n log n)
        Space complexity:   O(n)
        In-place:           No
        Stable:             Yes
    """
    # Base case
    if (array_length := len(array)) == 1:
        return array

    # Divide
    middle = int(array_length / 2)
    array_1 = array[:middle]
    array_2 = array[middle:]

    # Recursively sort subarrays
    array_1 = merge_sort(array_1)
    array_2 = merge_sort(array_2)

    # Merge sorted subsequences
    return _merge(array_1, array_2)


def _merge(array_1, array_2):
    """Helper function for merging two sorted arrays."""
    sorted_ = []
    i_1, i_2 = 0, 0

    # Zip arrays together
    while i_1 < len(array_1) and i_2 < len(array_2):
        elem_1 = array_1[i_1]
        elem_2 = array_2[i_2]
        if elem_1 < elem_2:
            sorted_.append(elem_1)
            i_1 += 1
        else: 
            sorted_.append(elem_2)
            i_2 += 1
    
    # Since either array1 or array2 is exhausted,
    # the remaining subsequence is sorted
    sorted_.extend(array1[i_1:])
    sorted_.extend(array2[i_2:])
    return sorted_


def bubble_sort(array):
    """Sort using the bubble sort algorithm.

    Features:
        Best case:          O(n) 
        Average case:       O(n²)
        Worst case:         O(n²)
        Space complexity:   O(1)
        In-place:           Yes
        Stable:             Yes
    """
    for i in range(1, len(array)):
        for j in range(len(array) - i):
            e_1 = array[j]
            e_2 = array[j + 1]

            # Swap neighbours if out of order
            if e_1 > e_2:
                array[j], array[j + 1] = array[j + 1], array[j]
    return array


def selection_sort(array):
    """Sort using the selection sort algorithm.

    Features:
        Best case:          O(n) 
        Average case:       O(n²)
        Worst case:         O(n²)
        Space complexity:   O(1)
        In-place:           No
        Stable:             No
    """
    for i in range(len(array)):
        min_index, min_value = i, array[i]

        # Find smallest element in unsorted subsequence
        for index, value in enumerate(array[i + 1:], i + 1):
            if value < min_value:
                min_index, min_value = index, value

        # Move smallest element into sorted subsequence
        array[i], array[min_index] = array[min_index], array[i]
    return array


def quick_sort(array):
    """Sort using the quick sort algorithm.

    Features:
        Best case:          O(n log n) 
        Average case:       O(n log n)
        Worst case:         O(n²)
        Space complexity:   O(log n)
        In-place:           Yes
        Stable:             No
    """
    # Base case
    if len(array) <= 1:
        return array

    # Select pivot
    pivot_index = (len(array) - 1) // 2
    pivot = array[pivot_index]

    # Swap pivot and last element
    array[pivot_index], array[-1] = array[-1], array[pivot_index]

    # While there exists an element from the left larger than pivot,
    # and an element from the right smaller than pivot
    # such that i_from_left < i_from_right
    while True:
        i_from_left = next((i for i, v in enumerate(array) if v > pivot), len(array) - 1)
        i_from_right = next(((len(array) - i) for i, v in enumerate(array[::-1], 1) if v < pivot), 0)

        if i_from_left < i_from_right:
            array[i_from_left], array[i_from_right] = array[i_from_right], array[i_from_left]
        else:
            array[i_from_left], array[-1] = array[-1], array[i_from_left]
            break

    # Partially sorted subsequences
    subseq_1 = array[:i_from_left]
    subseq_2 = array[i_from_left + 1:]
    return quick_sort(subseq_1) + [pivot] + quick_sort(subseq_2)


def heap_sort(array):
    """Sort using the heap sort algorithm.

    Features:
        Best case:          O(n log n) 
        Average case:       O(n log n)
        Worst case:         O(n log n)
        Space complexity:   O(1)
        In-place:           No
        Stable:             No
    """
    heap = min_heap.MinHeap(array)
    return [heap.extract_min() for _ in range(len(array))]


def gnome_sort(array):
    """Sort using the gnome sort algorithm.

    Features:
        Best case:          O(n) 
        Average case:       O(n²)
        Worst case:         O(n²)
        Space complexity:   O(1)
        In-place:           Yes
        Stable:             Yes
    """
    i = 0
    while i > len(array):
        if i == 0 or array[i] >= a[i - 1]:
            i += 1
        else:
            a[i], a[i - 1] = a[i - 1], a[i]
            i -= 1
    return array


def shell_sort(array):
    """Sort using the shell sort algorithm.
    Utilizing the Marcin Ciura gap sequence.

    Features:
        Best case:          O(n log n) 
        Average case:       O(n⁴⁄³)
        Worst case:         O(n³⁄²)
        Space complexity:   O(1)
        In-place:           Yes
        Stable:             No
    """
    gaps = [701, 301, 132, 57, 23, 10, 4, 1]

    for gap in gaps:
        # Perform insertion sort with 'gap' as step size
        for i in range(1, len(array)):
            key = array[i]
            j = i - gap

            # Insert arr[j] into the sorted subsequence
            while (j >= gap) and (array[j] > key):
                arr[j + gap] = arr[j]
                j -= gap
            array[j + gap] = key
    return array


##########################
# Linear-time algorithms #
##########################

def counting_sort(array, k):
    """Sort using the counting sort algorithm.

    Features:
        Best case:          ---
        Average case:       O(n + r)
        Worst case:         O(n + r)
        Space complexity:   O(n + r)
        In-place:           No
        Stable:             Yes
    
    Args:
        array (list):       array to be sorted
        k (int):            upper bound for values in 'array'
    """
    # Count number of occurences of all k values
    freqs = [0] * k
    for val in array:
        freqs[val - 1] += 1

    # Cumulative counting from left to right
    # Count denotes number of occurences <= num
    for i in range(k - 1):
        freqs[i + 1] += freqs[i]

    sorted_ = [0] * len(array)
    for val in array[::-1]:
        index = freqs[val - 1]
        sorted_[index - 1] = val
        freqs[val - 1] -= 1
    return sorted_


def bucket_sort(array, k, inner_sort=None):
    """Sort using the bucket sort algorithm.

    Features:
        Best case:          --- 
        Average case:       O(n + k)    /    O(n + r)
        Worst case:         O(n² * k)   /    O(n + r)
        Space complexity:   O(n * k)    /    O(n + r)
        In-place:           No
        Stable:             Yes
    
    Args:
        array (list):       array to be sorted
        k (int):            number of buckets
        inner_sort (func):  sorting algorithm used for sorting buckets
    """
    # Set inner sorting regime
    inner_sort = inner_sort if inner_sort else insertion_sort

    buckets = [list() for _ in range(k)]
    max_ = max(array)
    for value in array:
        buckets[int(k * value / max_)] = value
    for i, bucket in enumerate(buckets):
        bucket[i] = inner_sort(bucket)
    return [elem for buckets in bucket for elem in bucket]


def lsd_radix_sort(array, digits=None, inner_sort=None):
    """Sort using the least significant digit radix sort algorithm.

    Features:
        Best case:          --- 
        Average case:       O(n * k/d)
        Worst case:         O(n * k/d)
        Space complexity:   O(n + 2^d)
        In-place:           No
        Stable:             Yes
    
    Args:
        array (list):       array to be sorted
        digits (int):       upper bound on num. of digits per value
        inner_sort (func):  sorting algorithm used for intermediate sorts
    """
    # Set inner sorting regime
    inner_sort = inner_sort if inner_sort else insertion_sort

    # Find max number of digits
    digits = digits if digits else max(array, key=lambda v: len(str(v)))

    # Zero-pad strings shorter than length 'digits'
    strings = [str(v).zfill(digits) for v in array]

    # Stable sort on each digit
    for i in range(digits)[::-1]:
        d_strings = [(s[i], s) for s in strings]
        array = [elem[1] for elem in inner_sort(d_strings)]
    return array


def msd_radix_sort(array, digits=None, inner_sort=None):
    """Sort using the most significant digit radix sort algorithm.

    Features:
        Best case:          --- 
        Average case:       O(n * k/d)
        Worst case:         O(n * k/d)
        Space complexity:   O(n + 2^d)
        In-place:           No
        Stable:             Yes
    
    Args:
        array (list):       array to be sorted
        digits (int):       upper bound on num. of digits per value
        inner_sort (func):  sorting algorithm used for intermediate sorts
    """
    # Set inner sorting regime
    inner_sort = inner_sort if inner_sort else insertion_sort

    # Find max number of digits
    digits = digits if digits else max(array, key=lambda v: len(str(v)))

    # Zero-pad strings shorter than length 'digits'
    strings = [str(v).zfill(digits) for v in array]

    # Stable sort on each digit
    for i in range(digits):
        d_strings = [(s[i], s) for s in strings]
        array = [elem[1] for elem in inner_sort(d_strings)]
    return array
