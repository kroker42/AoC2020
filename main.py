# coding=utf-8

from itertools import permutations
import operator
import functools
import time
import unittest

# read one int per line from file
def read_ints(fname):
    return [int(line) for line in open(fname)]

# read lines from file
def read_lines(fname):
    return [line for line in open(fname)]

# Day 1

# get map of pair sum -> pair for all permutation pairs in a list
def get_pair_sums(array):
    return {p[0] + p[1] : p for p in permutations(array, 2)}

# find 3 numbers that add up to 2020 in expenses, using map of pair sums
def find_2020_sum3(expenses, pair_sums):
    for i in expenses:
        if 2020 - i in pair_sums:
            return pair_sums[2020 - i] + (i,)

class Day1Test(unittest.TestCase):
    def test_get_pair_sums(self):
        pairs = get_pair_sums([1721, 979, 366, 299, 675, 1456])
        self.assertEqual(pairs[2020], (299, 1721))

    def test_find_2020_sum3(self):
        test_set = [1721, 979, 366, 299, 675, 1456]
        pairs = get_pair_sums(test_set)
        self.assertEqual(find_2020_sum3(test_set, pairs), (675, 366, 979))

# find pairs and triplets in expenses that add up to 2020
def day1():
    expenses = read_ints('day1input.txt')

    start_time = time.time()

    pair_sums = get_pair_sums(expenses)
    task1 = functools.reduce(operator.mul, pair_sums[2020], 1)
    task2 = functools.reduce(operator.mul, find_2020_sum3(expenses, pair_sums), 1)

    return time.time() - start_time, task1, task2

# Day 2

# "1-5 a: yawtrdiop" -> [1, 5, "a", "yawtrdiop"]
def parse(pwd):
    items = pwd.split()
    return [int(x) for x in items[0].split("-")] + [items[1][0], items[2]]

# valid pwd acc. to sled rental policy
def is_valid_sled_rental(pwd):
    return pwd[3].count(pwd[2]) in range(pwd[0], pwd[1]+1)

#valid pwd acc. to Toboggan Corporate
def is_valid_toboggan(pwd):
    return sum([pwd[3][pwd[0]-1] == pwd[2], pwd[3][pwd[1]-1] == pwd[2]]) == 1

class Day2Test(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse("1-3 a: awopqtry"), [1, 3, "a", "awopqtry"])

    def test_is_valid_sled_rental(self):
        self.assertFalse(is_valid_sled_rental([2, 3, "s", "wdaas"]))
        self.assertTrue(is_valid_sled_rental(parse("1-3 a: abcde")))
        self.assertFalse(is_valid_sled_rental(parse("1-3 b: cdefg")))
        self.assertTrue(is_valid_sled_rental(parse("2-9 c: ccccccccc")))

    def test_is_valid_toboggan(self):
        self.assertTrue(is_valid_toboggan(parse("1-3 a: abcde")))
        self.assertFalse(is_valid_toboggan(parse("1-3 b: cdefg")))
        self.assertFalse(is_valid_toboggan(parse("2-9 c: ccccccccc")))


def day2():
    pwds = [parse(pwd) for pwd in read_lines('day2input.txt')]

    start_time = time.time()
    task1 = sum([is_valid_sled_rental(pwd) for pwd in pwds])
    task2 = sum([is_valid_toboggan(pwd) for pwd in pwds])

    return time.time() - start_time, task1, task2

# Day 3
#given a start index, is the position at 3 along a tree ('#'), using wrapped lines?
def is_tree(row, index):
    return row[index] == '#'

def trajectory(slope, step):
    trees = []
    row_len = len(slope[0])
    index = 0
    for row in slope[1:]:
        index = (index + step) % row_len
        trees.append(is_tree(row, index))
    return trees


class Day3Test(unittest.TestCase):
    tree_map = ['..##.......',
                '#...#...#..',
                '.#....#..#.',
                '..#.#...#.#',
                '.#...##..#.',
                '..#.##.....',
                '.#.#.#....#',
                '.#........#',
                '#.##...#...',
                '#...##....#',
                '.#..#...#.#']
    slope = [
    '....#...#####..##.#..##..#....#',  #0
    '..##.#.#.........#.#......##...',
    '#.#.#.##.##...#.......#...#..#.',
    '..##.............#.#.##.....#..',
    '##......#.............#....#...',
    '.....##..#.....##.#.......##..#',
    '.##.....#........##...##.#....#',
    '.##......#.#......#.....#..##.#',
    '##....#..#...#...#...##.#...##.',
    '##........##.#...##......#.#.#.',
    '..#.#........#...##.....#.....#',  #10
    '..#.......####.#....#..#####...',
    '.##..#..#..##.#.....###.#..#...',
    '......###..##.....#.#.#..###.#.',  #13
    '..#.#...#..##.....#....#.#.....',
    '.....# .#...#.###.#..#..........',
    '##.....#...#.#....#..#.#.......',
    '..#...#...#.........##......#..',
    '......#.#...#...#..#...##.#...#',
    '....#.................##.##....',
    '...#......#.............#....##',
    '##..#..#..........#...##.#.#...',
    '....#...##....#..#.#...........',
    '##.#.#.#...#....#........#..#.#',
    '...###..........#...#...#..##.#',
    '..##.......###.#......##.##....',
    '...........#.#....#.....#.#...#',
    '..#......##.#...##.#.#......#.#']

    def test_is_tree(self):
        self.assertTrue(is_tree(self.tree_map[0], 3))
        self.assertTrue(is_tree(self.tree_map[1], 0))
        self.assertFalse(is_tree(self.tree_map[1], 3))

    def test_trajectory(self):
        # task 1
        self.assertEqual(7, sum(trajectory(self.tree_map, 3)))
        self.assertEqual(31, len(self.slope[0]))
        self.assertEqual(8, sum(trajectory(self.slope[0:12], 3)))
        self.assertEqual(10, sum(trajectory(self.slope[0:14], 3)))
        # task 2
        self.assertEqual(2, sum(trajectory(self.tree_map, 1)))
        self.assertEqual(3, sum(trajectory(self.tree_map, 5)))
        self.assertEqual(4, sum(trajectory(self.tree_map, 7)))
        self.assertEqual(2, sum(trajectory(self.tree_map[::2], 1)))



def day3():
    slope = [line for line in read_lines('day3input.txt')]
    slope = [row[:-1] for row in slope]

    start_time = time.time()
    task1 = sum(trajectory(slope, 3))
    task2 = task1 * sum(trajectory(slope, 1)) * sum(trajectory(slope, 5)) * sum(trajectory(slope, 7)) * sum(trajectory(slope[::2], 1))

    return time.time() - start_time, task1, task2

def run(day):
    run_time, task1, task2 = day()
    print(day.__name__ + ": %.6s s - " % run_time + str(task1) + " " + str(task2))

if __name__ == '__main__':
    run(day1)
    run(day2)
    run(day3)
    unittest.main()

