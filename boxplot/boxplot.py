import matplotlib.pyplot as plt
from collections.abc import Iterable
from numbers import Number
import math

def percentile(n, percentile):
    index = percentile * (len(n) - 1) / 100
    if (len(n) - 1) % 100 == 0:
        return sorted(n)[int(index)]
    else:
        lower = sorted(n)[int(index)]
        upper = sorted(n)[int(index) + 1]
        interpolation = index - int(index)
        return lower + (upper - lower) * interpolation

def compute_stats(numbers):
    numbers = sorted(numbers)  # It's better to sort the numbers only once
    min_num = numbers[0]
    max_num = numbers[-1]
    p25 = percentile(numbers, 25)
    p50 = percentile(numbers, 50)
    p75 = percentile(numbers, 75)

    return min_num, max_num, p25, p50, p75

# returns boxplot from 1 series in a list of size 'width'
def _boxplot_chart(numbers, mn, mx, width=80):
    if not (width > 0):
        # TODO: raise something
        return '' 
    scale = lambda v: max(0, min(width - 1, int(width * 1.0 * (v - mn) / (mx - mn))))
    res = [' '] * width
    if len(numbers) == 0:
        return res
    if len(numbers) == 1:
        res[scale(numbers[0])] = '|'
        return res

    min_num, max_num, p25, p50, p75 = compute_stats(numbers)
    p25i = scale(p25)
    p75i = scale(p75)
    p50i = scale(p50)
    min_i = scale(min_num)
    max_i = scale(max_num)

    for i in range(min_i + 1, p25i):
        res[i] = '-'
    for i in range(p75i + 1, max_i):
        res[i] = '-'
    for i in range(p25i + 1, p75i):
        res[i] = '='
    
    res[min_i] = '|'
    res[max_i] = '|'

    if p50i == p25i and p50i == p75i:
        # 1 symbol for 3 marks. Use median
        res[p50i] = ':'
    else:
        # we have at least 2 symbols for 3 marks. In this case, p50 might get overwritten
        res[p50i] = ':'
        res[p25i] = '['
        res[p75i] = ']'

    return res

# returns line together with title, boundaries and chart
def _boxplot_line(numbers, mn, mx, title='', chart_width=80, left_margin=20):
    chart = _boxplot_chart(numbers, mn, mx, width=chart_width)
    if title != '':
        title = f'{title}: '
    #left = f'{title}{mn:.3g}('[-left_margin:]
    #left = f"{left:>{left_margin}}"
    #right = f'){mx:<3g}'
    
    left = f'{title}'[-left_margin:]
    left = f"{left:>{left_margin}}"
    right = ''
    return left + ''.join(chart) + right

def boxplot(numbers, mn, mx, title='', width=80):
    return _boxplot_line(numbers, mn, mx, title=title)

def boxplots(numbers, chart_width=80):
    mn = min(min(v) for v in numbers.values())
    mx = max(max(v) for v in numbers.values())
    for title, values in numbers.items():
        print(_boxplot_line(values, mn, mx, title=title))

if __name__ == '__main__':
    # plt.boxplot([]) <- empty
    # plt.boxplot([1]) <- single line, no box/whiskers
    # plt.boxplot([1,2]) <- whiskers at 1, 2, box interpolated
    # plt.boxplot([1,2,3]) <- whiskers at 1, 3, box interpolated, median at 2
    #plt.boxplot([1, 2, 100])
    #plt.show()
    data = {
        'A': [1, 2,3 ,4, 100],
        'B': [-10, -5, -1, -2, -3, -1],
        'C': [10, 20, 30, 40, 50, 60]
    }
    boxplots(data)
    print(boxplot([1,2], 0, 4, title='2 poindsjkfhkajsdhflkajsdhflkasdjhfts'))