from fewlines import charts as fc
import numpy as np

A = np.random.normal(size=100)
print('Just the list of numbers')
for l in fc.histogram_chart(A):
    print(l)
print('the list of numbers with extra options')
for l in fc.histogram_chart((A, {'n_lines': 3})):
    print(l)
print('the dict title -> numbers')
for l in fc.histogram_chart({'series_A': A}):
    print(l)
print('the dict title -> numbers, options')
for l in fc.histogram_chart({'series_A': (A, {'n_lines': 3})}):
    print(l)