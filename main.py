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

# read multiple-line records from file separated by empty line
def read_records(fname):
    return [record for record in open(fname).read().split('\n\n')]

# Day 1 - Submit expenses

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

# Day 2 - Validate Toboggan Corporate password policies

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

# Day 3 - Toboggan down the slope

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

# Day 4 - Hack the passport scanner
def parse_dict(str):
    return {k: v for k, v in [kv.split(':') for kv in str.split()]}

# is real passport or is North Pole Credentials - with cid field missing
def is_valid_passport(pp):
    return len(pp) == 8 or (len(pp) == 7 and "cid" not in pp.keys())

# height is e.g. "156cm" or "62in", and in the valid range for each height unit
def is_valid_height(str):
    restrictions = {"cm": range(150, 194), "in": range(59, 77)}
    if str[-2:] in restrictions.keys():
        height = int(re.search(r'\d+', str).group())
        return height and height in restrictions[str[-2:]]

    return False

# a '#' followed by exactly six characters 0-9 or a-f.
def is_valid_hair(str):
    return re.search(r"#[0-9a-f]{6}", str) != None

# bizarrely slower than the string comparison below
def is_valid_pid(str):
    return len(str) == 9 and re.match(r"^\d{9}$", str) != None

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
    data = ["ecl:gry pid:860033327 eyr:2020 hcl:#fffffd byr:1937 iyr:2017 cid:147 hgt:183cm",
            "iyr:2013 ecl:amb cid:350 eyr:2023 pid:028048884 hcl:#cfa07d byr:1929",
            "hcl:#ae17e1 iyr:2013 eyr:2024 ecl:brn pid:760753108 byr:1931 hgt:179cm",
            "hcl:#cfa07d eyr:2025 pid:166559648 iyr:2011 ecl:brn hgt:59in"]

    def test_is_valid_passport(self):
        exp_results = [True, False, True, False]
        pps = [parse_dict(pp) for pp in self.data]
        self.assertEqual(len(exp_results), len(pps))

        for i in range(0, len(pps)):
            self.assertEqual(exp_results[i], is_valid_passport(pps[i]))

    def test_is_valid_height(self):
        self.assertTrue(is_valid_height("170cm"))
        self.assertTrue(is_valid_height("60in"))
        self.assertFalse(is_valid_height("123cm"))
        self.assertFalse(is_valid_height("123in"))
        self.assertFalse(is_valid_height("123"))

    def test_is_valid_hair(self):
        self.assertTrue(is_valid_hair("#12de4f"))
        self.assertFalse(is_valid_hair("12de4f"))
        self.assertFalse(is_valid_hair("#z2de4f"))
        self.assertFalse(is_valid_hair("#e4f"))

    def test_is_valid_pid(self):
        self.assertTrue(is_valid_pid("012345678"))

    invalid_pps = ["eyr:1972 cid:100 hcl:#18171d ecl:amb hgt:170 pid:186cm iyr:2018 byr:1926",
                   "iyr:2019 hcl:#602927 eyr:1967 hgt:170cm ecl:grn pid:012533040 byr:1946",
                   "hcl:dab227 iyr:2012 ecl:brn hgt:182cm pid:021572410 eyr:2020 byr:1992 cid:277",
                   "hgt:59cm ecl:zzz eyr:2038 hcl:74454a iyr:2023 pid:3556412378 byr:2007"]

    valid_pps = ["pid:087499704 hgt:74in ecl:grn iyr:2012 eyr:2030 byr:1980 hcl:#623a2f",
                 "eyr:2029 ecl:blu cid:129 byr:1989 iyr:2014 pid:896056539 hcl:#a97842 hgt:165cm",
                 "hcl:#888785 hgt:164cm byr:2001 iyr:2015 cid:88 pid:545766238 ecl:hzl eyr:2022",
                 "iyr:2010 hgt:158cm hcl:#b6652a ecl:blu byr:1944 eyr:2021 pid:093154719"]

    def test_has_valid_fields(self):
        self.assertEqual(len(self.valid_pps), sum([has_valid_fields(pp) for pp in [parse_dict(d) for d in self.valid_pps]]))
        self.assertEqual(0, sum([has_valid_fields(pp) for pp in [parse_dict(d) for d in self.invalid_pps]]))

def day4():
    data = [parse_dict(record) for record in read_records('day4input.txt')]

    start_time = time.time()

    valid_passports = [pp for pp in data if is_valid_passport(pp)]
    task1 = len(valid_passports)
    task2 = sum([has_valid_fields(pp) for pp in valid_passports])

    return time.time() - start_time, task1, task2

# Day 5: Binary boarding cards
def binary(bin_str):
    valid_chars = ['B', 'R']
    return int(''.join([str(int(ch in valid_chars)) for ch in bin_str]), 2)

def get_seat(boarding_card):
    row = binary(boarding_card[:7])
    col = binary(boarding_card[7:10])
    return 8 * row + col

class Day5Test(unittest.TestCase):
    def test_binary(self):
        self.assertEqual(44, binary("FBFBBFF"))
        self.assertEqual(70, binary("BFFFBBF"))
        self.assertEqual(5, binary("RLR"))

    def test_seat_no(self):
        self.assertEqual(357, get_seat("FBFBBFFRLR"))
        self.assertEqual(567, get_seat("BFFFBBFRRR"))
        self.assertEqual(119, get_seat("FFFBBBFRRR"))
        self.assertEqual(820, get_seat("BBFFBBFRLL"))

def day5():
    boarding_cards = read_lines('day5input.txt')

    start_time = time.time()

    seats = [get_seat(card) for card in boarding_cards]
    task1 = max(seats)

    seats.sort()
    task2 = set(range(seats[0], seats[-1] + 1)).difference(seats)

    return time.time() - start_time, task1, task2

# Day 6 - Customs forms
def day6():
    groups = [g.split() for g in read_records('day6input.txt')]

    start_time = time.time()

    # remove line separators inside groups, then count unique items
    task1 = sum([len(set(''.join(g))) for g in groups])

    # count the chars that are the same in every line in a group
    task2 = sum([len(set(g[0]).intersection(*[set(x) for x in g[1:]])) for g in groups])

    return time.time() - start_time, task1, task2

# Luggage sorting
def parse_bags(str):
    bags = re.findall(r"(\w+\s\w+)\sbag", str)
    quantities = [int(i) for i in re.findall(r"\d+", str)]

    return bags, quantities

def build_tree(rules):
    tree = {}
    for r in rules:
        bags, quantities = r
        tree[bags[0]] = {}
        if quantities:
            tree[bags[0]] = dict(zip(bags[1:], quantities))
    return tree

def build_reverse_tree(rules):
    tree = {}
    for r in rules:
        bags, quantities = r
        if bags[0] not in tree:
            tree[bags[0]] = {}
        for i in range(0, len(quantities)):
            if bags[i + 1] not in tree:
                tree[bags[i + 1]] = {}
            tree[bags[i + 1]][bags[0]] = quantities[i]
    return tree

def reverse(tree):
    reverse_tree = {}
    for bag in tree.keys():
        if bag not in reverse_tree:
            reverse_tree[bag] = {}
        for b in tree[bag]:
            if b not in reverse_tree:
                reverse_tree[b] = {}
            reverse_tree[b][bag] = tree[bag][b]
    return reverse_tree

def build_map(rules):
    tree = build_tree(rules)
    return tree, reverse(tree)

def find_containers(reverse_tree, bag):
    containers = set(reverse_tree[bag].keys())
    for b in containers:
        containers = containers.union(find_containers(reverse_tree, b))
    return containers

def find_content(tree, bag):
    count = 0
    for b in tree[bag].keys():
        count += tree[bag][b] * (1 + find_content(tree, b))
    return count

class Day7Test(unittest.TestCase):
    rules = [
        "light red bags contain 1 bright white bag, 2 muted yellow bags.",
        "dark orange bags contain 3 bright white bags, 4 muted yellow bags.",
        "bright white bags contain 1 shiny gold bag.",
        "muted yellow bags contain 2 shiny gold bags, 9 faded blue bags.",
        "shiny gold bags contain 1 dark olive bag, 2 vibrant plum bags.",
        "dark olive bags contain 3 faded blue bags, 4 dotted black bags.",
        "vibrant plum bags contain 5 faded blue bags, 6 dotted black bags.",
        "faded blue bags contain no other bags.",
        "dotted black bags contain no other bags."]

    def test_parse_bags(self):
        self.assertEqual(["light red", "bright white", "muted yellow"], parse_bags(self.rules[0])[0])
        self.assertEqual([1, 2], parse_bags(self.rules[0])[1])
        self.assertEqual(["dotted black", "no other"], parse_bags(self.rules[-1])[0])
        self.assertEqual([], parse_bags(self.rules[-1])[1])

    def test_build_tree(self):
        parsed = [parse_bags(r) for r in self.rules]
        tree = build_tree(parsed[-3:])
        exp_tree = {"faded blue": {}, "dotted black": {},
                    "vibrant plum": {"faded blue": 5, "dotted black": 6}}
        self.assertEqual(exp_tree, tree)

    def test_build_reverse_tree(self):
        parsed = [parse_bags(r) for r in self.rules]
        reverse_tree = build_reverse_tree(parsed[-3:])
        exp_tree = {"faded blue": {"vibrant plum": 5},
                    "dotted black": {"vibrant plum": 6},
                    "vibrant plum": {}}
        self.assertEqual(exp_tree, reverse_tree)

    def test_find_containers(self):
        self.assertEqual(4, len(find_containers(build_reverse_tree([parse_bags(r) for r in self.rules]), "shiny gold")))

    rules2 = [
    "shiny gold bags contain 2 dark red bags.",
    "dark red bags contain 2 dark orange bags.",
    "dark orange bags contain 2 dark yellow bags.",
    "dark yellow bags contain 2 dark green bags.",
    "dark green bags contain 2 dark blue bags.",
    "dark blue bags contain 2 dark violet bags.",
    "dark violet bags contain no other bags."
    ]

    def test_find_content(self):
        self.assertEqual(32, find_content(build_tree([parse_bags(r) for r in self.rules]), "shiny gold"))
        self.assertEqual(126, find_content(build_tree([parse_bags(r) for r in self.rules2]), "shiny gold"))

    def test_build_map(self):
        parsed = [parse_bags(r) for r in self.rules2]
        tree, rev_tree = build_map(parsed)
        exp_tree = {
            "shiny gold" : {},
            "dark red": {"shiny gold": 2},
            "dark orange": {"dark red": 2},
            "dark yellow": {"dark orange": 2},
            "dark green": {"dark yellow": 2},
            "dark blue": {"dark green": 2},
            "dark violet": {"dark blue": 2},
        }
        self.assertEqual(exp_tree, rev_tree)

def day7():
    rules = read_lines("day7input.txt")

    start_time = time.time()
    parsed_rules = [parse_bags(r) for r in rules]
    tree, rev_tree = build_map(parsed_rules)
    task1 = len(find_containers(rev_tree, "shiny gold"))
    task2 = find_content(tree, "shiny gold")

    return time.time() - start_time, task1, task2

# Main
def run(day):
    run_time, task1, task2 = day()
    print(day.__name__ + ": %.6s s - " % run_time + str(task1) + " " + str(task2))

if __name__ == '__main__':
    for i in range(1, 8):
        run(eval("day" + str(i)))
    unittest.main()

