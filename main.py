# coding=utf-8

from itertools import permutations
import operator
import functools
import time

# read one int per line from file
def read_ints(fname):
    return [int(line) for line in open(fname)]

# read lines from file
def read_lines(fname):
    return [line for line in open(fname)]

# get map of pair sum -> pair for all permutation pairs in a list
def get_pair_sums(array):
    return {p[0] + p[1] : p for p in permutations(array, 2)}

def test_get_pair_sums():
    pairs = get_pair_sums([1721, 979, 366, 299, 675, 1456])
    result = (1721, 299)
    print("test_find_2020_sum: " + str(result) + str(pairs[2020]))

# find 3 numbers that add up to 2020 in expenses, using map of pair sums
def find_2020_sum3(expenses, pair_sums):
    for i in expenses:
        if 2020 - i in pair_sums:
            return pair_sums[2020 - i] + (i,)

def test_find_2020_sum3():
    test_set = [1721, 979, 366, 299, 675, 1456]
    pairs = get_pair_sums(test_set)
    result = (979, 366, 675)
    print("test_find_2020_sum3: " + str(result) + str(find_2020_sum3(test_set, pairs)))

# find pairs and triplets in expenses that add up to 2020
def day1():
    expenses = read_ints('day1input.txt')
    start_time = time.time()
    pair_sums = get_pair_sums(expenses)
    task1 = functools.reduce(operator.mul, pair_sums[2020], 1)
    task2 = functools.reduce(operator.mul, find_2020_sum3(expenses, pair_sums), 1)

    print("Day 1 --- %s seconds ---" % (time.time() - start_time))
    print(task1)
    print(task2)

def parse(pwd):
    items = pwd.split()
    itemised = items[0].split("-") + [items[1][0], items[2]]
    return itemised

def test_parse():
    print("test_parse: ")
    print(*parse("1-3 a: awopqtry"))

# valid pwd acc. to sled rental policy
def is_valid_sled_rental(pwd):
    return pwd[3].count(pwd[2]) in range(int(pwd[0]), int(pwd[1])+1)

def test_is_valid_sled_rental():
    print("test_is_valid:" + str(is_valid_sled_rental(["1", "3", "a", "wdaas"])))
    print("test_is_valid:" + str(not is_valid_sled_rental(["1", "3", "q", "wdaas"])))
    print("test_is_valid:" + str(not is_valid_sled_rental(["2", "3", "s", "wdaas"])))

#valid pwd acc. to Toboggan Corporate
def is_valid_toboggan(pwd):
    return sum([pwd[3][int(pwd[0])-1] == pwd[2], pwd[3][int(pwd[1])-1] == pwd[2]]) == 1

def test_is_valid_toboggan():
    print("test_is_valid_toboggan:" + str(is_valid_toboggan(parse("1-3 a: abcde"))))
    print("test_is_valid_toboggan:" + str(not is_valid_toboggan(parse("1-3 b: cdefg"))))
    print("test_is_valid_toboggan:" + str(not is_valid_toboggan(parse("2-9 c: ccccccccc"))))

def day2():
    print("Day 2")
    pwds = [parse(pwd) for pwd in read_lines('day2input.txt')]
    print(pwds)

    start_time = time.time()
    task1 = [is_valid_sled_rental(pwd) for pwd in pwds].count(True)
    task2 = [is_valid_toboggan(pwd) for pwd in pwds].count(True)

    print("--- %s seconds ---" % (time.time() - start_time))
    print(task1)
    print(task2)


def test():
    test_get_pair_sums()
    test_find_2020_sum3()
    test_parse()
    test_is_valid_sled_rental()
    test_is_valid_toboggan()

if __name__ == '__main__':
    test()
    day1()
    day2()

