# coding=utf-8

from itertools import combinations
import operator
import functools
import re

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
    return {p[0] + p[1] : p for p in combinations(array, 2)}

# find 3 numbers that add up to 2020 in expenses, using map of pair sums
def find_2020_sum3(expenses, pair_sums):
    for i in expenses:
        if 2020 - i in pair_sums:
            return pair_sums[2020 - i] + (i,)

class Day1Test(unittest.TestCase):
    def test_get_pair_sums(self):
        pairs = get_pair_sums([1721, 979, 366, 299, 675, 1456])
        self.assertEqual(pairs[2020], (1721, 299))

    def test_find_2020_sum3(self):
        test_set = [1721, 979, 366, 299, 675, 1456]
        pairs = get_pair_sums(test_set)
        self.assertEqual(find_2020_sum3(test_set, pairs), (366, 675, 979))

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

def trajectory(slope, step):
    row_len = len(slope[0])
    index = 0
    count = 0

    for row in slope[1:]:
        index = (index + step) % row_len
        count += row[index] == '#'
    return count

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

    def test_trajectory(self):
        # task 1
        self.assertEqual(7, trajectory(self.tree_map, 3))
        # task 2
        self.assertEqual(2, trajectory(self.tree_map, 1))
        self.assertEqual(3, trajectory(self.tree_map, 5))
        self.assertEqual(4, trajectory(self.tree_map, 7))
        self.assertEqual(2, trajectory(self.tree_map[::2], 1))



def day3():
    slope = [row[:-1] for row in read_lines('day3input.txt')]

    start_time = time.time()
    task1 = trajectory(slope, 3)
    task2 = task1 * trajectory(slope, 1) * trajectory(slope, 5) * trajectory(slope, 7) * trajectory(slope[::2], 1)

    return time.time() - start_time, task1, task2

# Day 4
def parse_passports(data):
    passports = []
    pp = ""

    for line in data:
        if not line:
            passports.append(pp)
            pp = line
        else:
            pp += line + " "
    passports.append(pp)
    return passports

def parse_dict(str):
    result = {k: v for k, v in [kv.split(':') for kv in str.split()]}
    return result

# is real passport or is North Pole Credentials - with cid field missing
def is_valid_passport(pp):
    return len(pp) == 8 or (len(pp) == 7 and "cid" not in pp.keys())

def is_valid_height(str):
    height = None
    if str[-2:] == "cm" or str[-2:] == "in":
        height = int(re.search(r'\d+', str).group())
    if height:
        if str[-2:] == "cm":
            return height in range(150, 194)
        else:
            return height in range(59, 77)

# a # followed by exactly six characters 0-9 or a-f.
def is_valid_hair(str):
    if str[0] == '#':
        m = re.search(r"[0-9a-f]{6}", str)
        return m and len(m.group()) == 6
    else:
        return False

# for a valid passport - are the fields also valid?
# pre-condition; is_valid_passport(pp)
def has_valid_fields(pp):
    return \
        int(pp['byr']) in range(1920, 2003) and \
        int(pp['iyr']) in range(2010, 2021) and \
        int(pp['eyr']) in range(2020, 2031) and \
        is_valid_height(pp['hgt']) and \
        is_valid_hair(pp['hcl']) and \
        pp['ecl'] in ["amb", "blu", "brn", "gry", "grn", "hzl", "oth"] and \
        len(pp['pid']) == 9 and pp['pid'].isdigit()

class Day4Test(unittest.TestCase):
    data = ["ecl:gry pid:860033327 eyr:2020 hcl:#fffffd",
                 "byr:1937 iyr:2017 cid:147 hgt:183cm",
                 "",
                 "iyr:2013 ecl:amb cid:350 eyr:2023 pid:028048884",
                 "hcl:#cfa07d byr:1929",
                 "",
                 "hcl:#ae17e1 iyr:2013",
                 "eyr:2024",
                 "ecl:brn pid:760753108 byr:1931",
                 "hgt:179cm",
                 "",
                 "hcl:#cfa07d eyr:2025 pid:166559648",
                 "iyr:2011 ecl:brn hgt:59in"]

    pps = [parse_dict(kv) for kv in parse_passports(data)]

    def test_parse_dict(self):
        self.assertEqual({"hcl": "#cfa07d", "byr": "1929"}, parse_dict(self.data[4]))

    def test_parse_passports(self):
        self.assertEqual(["ecl:gry pid:860033327 eyr:2020 hcl:#fffffd byr:1937 iyr:2017 cid:147 hgt:183cm ",
                          "iyr:2013 ecl:amb cid:350 eyr:2023 pid:028048884 hcl:#cfa07d byr:1929 "],
                         parse_passports(self.data[:5]))
        self.assertEqual(4, len(self.pps))

    def test_is_valid_passport(self):
        self.assertTrue(is_valid_passport(self.pps[0]))
        self.assertFalse(is_valid_passport(self.pps[1]))
        self.assertTrue(is_valid_passport(self.pps[2]))
        self.assertFalse(is_valid_passport(self.pps[3]))

    def test_is_valid_height(self):
        self.assertFalse(is_valid_height("123cm"))
        self.assertFalse(is_valid_height("123in"))
        self.assertFalse(is_valid_height("123"))
        self.assertTrue(is_valid_height("170cm"))
        self.assertTrue(is_valid_height("60in"))

    def test_is_valid_hair(self):
        self.assertTrue(is_valid_hair("#12de4f"))
        self.assertFalse(is_valid_hair("12de4f"))
        self.assertFalse(is_valid_hair("#z2de4f"))
        self.assertFalse(is_valid_hair("#e4f"))

    invalid_pps = ["eyr:1972 cid:100 hcl:#18171d ecl:amb hgt:170 pid:186cm iyr:2018 byr:1926",
                 "iyr:2019 hcl:#602927 eyr:1967 hgt:170cm ecl:grn pid:012533040 byr:1946",
                 "hcl:dab227 iyr:2012 ecl:brn hgt:182cm pid:021572410 eyr:2020 byr:1992 cid:277",
                 "hgt:59cm ecl:zzz eyr:2038 hcl:74454a iyr:2023 pid:3556412378 byr:2007"]

    valid_pps = ["pid:087499704 hgt:74in ecl:grn iyr:2012 eyr:2030 byr:1980 hcl:#623a2f",
                 "eyr:2029 ecl:blu cid:129 byr:1989 iyr:2014 pid:896056539 hcl:#a97842 hgt:165cm",
                 "hcl:#888785 hgt:164cm byr:2001 iyr:2015 cid:88 pid:545766238 ecl:hzl eyr:2022",
                 "iyr:2010 hgt:158cm hcl:#b6652a ecl:blu byr:1944 eyr:2021 pid:093154719"]

    def test_has_valid_fields(self):
        for pp in self.invalid_pps:
            self.assertFalse(has_valid_fields(parse_dict(pp)))

        for pp in self.valid_pps:
            self.assertTrue(has_valid_fields(parse_dict(pp)))

def day4():
    data = [row[:-1] for row in read_lines('day4input.txt')]

    start_time = time.time()
    passports = [parse_dict(kv) for kv in parse_passports(data)]
    valid_passports = [pp for pp in passports if is_valid_passport(pp)]
    task1 = len(valid_passports)
    task2 = 0
    for pp in valid_passports:
        if has_valid_fields(pp):
            task2 += 1

    return time.time() - start_time, task1, task2

def run(day):
    run_time, task1, task2 = day()
    print(day.__name__ + ": %.6s s - " % run_time + str(task1) + " " + str(task2))

if __name__ == '__main__':
    run(day1)
    run(day2)
    run(day3)
    run(day4)
    unittest.main()

