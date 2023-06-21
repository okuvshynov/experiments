import random
import timeit

# Generate a random dictionary of number-to-number mappings
def generate_random_dict(size):
    return {str(num): num for num in range(size)}

def benchmark_lookup(dictionary, size):
    existing_key = str(random.randint(0, size - 1))
    missing_key = str(size)
    iters = 1000000

    # Benchmark lookup for an existing key
    exist_double = timeit.timeit(lambda: dictionary[existing_key] if existing_key in dictionary else 0, number=iters)
    missing_double = timeit.timeit(lambda: dictionary[missing_key] if missing_key in dictionary else 0, number=iters)

    # Benchmark lookup for a missing key
    exist_single = timeit.timeit(lambda: dictionary.get(existing_key, 0), number=iters)
    missing_single = timeit.timeit(lambda: dictionary.get(missing_key, 0), number=iters)
    print(f'existing: {exist_double}, {exist_single}')
    print(f'missing: {missing_double}, {missing_single}')
    

# Generate a random dictionary of size 10000
dictionary_size = 100000
random_dict = generate_random_dict(dictionary_size)

# Benchmark lookup performance
benchmark_lookup(random_dict, dictionary_size)