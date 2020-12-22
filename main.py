# coding=utf-8

import functools
import operator
import re
import time
import unittest
import glob

from itertools import combinations
from itertools import permutations
from collections import deque
from math import prod


import np

from filereader import read_lines
from GameOfLife import day11
from GameOfLife import day17
from langs import day16
from langs import day21

# read one int per line from file
def read_ints(fname):
    return [int(line) for line in open(fname)]


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

# Infinite games loop

def split_instructions(code):
    return [(i, int(j)) for i, j in [l.split() for l in code]]

def execute(code, i, acc):
    if code[0] == "acc":
        i += 1
        acc += code[1]
    elif code[0] == "jmp":
        i += code[1]
    elif code[0] == "nop":
        i += 1

    return i, acc

def run_code(code, repair = None):
    visited = dict(zip(range(0, len(code)), [0] * len(code)))

    i = 0
    acc = 0

    while i < len(code) and not visited[i]:
        visited[i] = 1
        next_instr = repair[1] if repair and i == repair[0] else code[i]
        i, acc = execute(next_instr, i, acc)

    return i, acc

def repair(code):
    instr = None

    for i in range(0, len(code)):
        if code[i][0] == "jmp":
            instr = (i, ["nop", code[i][1]])
        elif code[i][0] == "nop":
            instr = (i, ["jmp", code[i][1]])

        next_i, acc = run_code(code, instr)
        if next_i == len(code):
            return acc

    return None

class Day8Test(unittest.TestCase):
    code = ["nop +0",
            "acc +1",
            "jmp +4",
            "acc +3",
            "jmp -3",
            "acc -99",
            "acc +1",
            "jmp -4",
            "acc +6"]
    instr = split_instructions(code)

    def test_run_code(self):
        self.assertEqual(5, run_code(self.instr)[1])

    def test_repair(self):
        self.assertEqual(8, repair(self.instr))

def day8():
    code = read_lines("day8input.txt")

    start_time = time.time()
    instr = split_instructions(code)
    task1 = run_code(instr)[1]
    task2 = repair(instr)

    return time.time() - start_time, task1, task2

# Hacking the plane

def check(code, preamble):
    sums = [sum(x) for x in permutations(code[:preamble], 2)]

    for i in range(preamble, len(code)):
        num = code[i]
        if num not in sums:
            return num
        else:
            sums = sums[preamble-1:] + [sum((x, num)) for x in code[i - (preamble - 1): i]]

def encryption_weakness(code, num):
    for i in range(0, len(code) - 1):
        sum = code[i]
        for j in range(i + 1, len(code)):
            sum += code[j]
            if sum == num:
                return i, j+1
            elif sum > num:
                break


def sum_up(code, num, j, sum):
    while sum < num and j < len(code):
        sum += code[j]
        j += 1
    return sum, j


def encryption_weakness2(code, num):
    i = 0
    j = 1
    sum = code[i]

    for i in range(0, len(code) - 1):
        sum, j = sum_up(code, num, j, sum)

        if sum == num:
            return i, j
        else:
            sum -= code[i]

class Day9Test(unittest.TestCase):
    code = [35,
            20,
            15,
            25,
            47,
            40,
            62,
            55,
            65,
            95,
            102,
            117,
            150,
            182,
            127,
            219,
            299,
            277,
            309,
            576]
    def test_get_sums(self):
        self.assertEqual(127, check(self.code, 5))


# encryption_weakness:  0.0137 s - 88311122 13549369
# encryption_weakness2: 0.0045 s - 88311122 13549369
def day9():
    code = read_ints("day9input.txt")

    start_time = time.time()

    task1 = check(code, 25)
    start, end = encryption_weakness2(code, task1)
    task2 = max(code[start: end]) + min(code[start: end])

    return time.time() - start_time, task1, task2

# Battery charger

def chain(adapters):
    adapters.append(0)
    adapters.sort()
    adapters.append(adapters[-1] + 3)
    diffs = [j - i for i, j in zip(adapters[:-1], adapters[1:])]
    return diffs

def tree_arrange(adapters):
    nodes = {0: []}
    for a in adapters[1:]:
        nodes[a] = []
        for l in nodes:
            diff = a - l
            if diff == 3:
                l.leaves.append(a)

def tree_arrange(adapter):
    options = {a: [] for a in adapter}
    leaves = []

    for a in adapter:
        diffs = [a - leaf for leaf in leaves]
        nodes = []
        for i in range(0, len(diffs)):
            if diffs[i] == 3:
                options[leaves[i]].append(a)
                nodes.append(leaves[i])
            elif diffs[i] > 3:
                nodes.append(leaves[i])
            else:
                options[leaves[i]].append(a)
        leaves.append(a)
        for n in nodes:
            leaves.remove(n)

    items = list(options.keys())
    items.reverse()

    counts = {items[0]: 1}

    for a in items[1:]:
        counts[a] = 0
        for n in options[a]:
            counts[a] += counts[n]

    return counts[0]


class Day9Test(unittest.TestCase):
    adapters = [16,
                10,
                15,
                5,
                1,
                11,
                7,
                19,
                6,
                12,
                4]

    adapters2 = [28,
            33,
            18,
            42,
            31,
            14,
            46,
            20,
            48,
            47,
            24,
            23,
            49,
            45,
            19,
            38,
            39,
            11,
            1,
            32,
            25,
            35,
            8,
            17,
            7,
            9,
            4,
            2,
            34,
            10,
            3]

    def test(self):
        diffs = chain(self.adapters2)
        self.assertEqual(0, self.adapters2[0])
        self.assertEqual(220, diffs.count(1) * diffs.count(3))

    def test_arrange(self):
        chain(self.adapters)
        self.assertEqual(8, tree_arrange(self.adapters))
        self.assertEqual(19208, tree_arrange(self.adapters2))


def day10():
    adapters = read_ints("day10input.txt")

    start_time = time.time()

    adapters.append(0)
    adapters.sort()
    adapters.append(adapters[-1] + 3)
    diffs = [j - i for i, j in zip(adapters[:-1], adapters[1:])]

    task1 = diffs.count(1) * diffs.count(3)
    task2 = tree_arrange(adapters)

    return time.time() - start_time, task1, task2

# Day 12 - move ship


directions = ['N', 'E', 'S', 'W']


def rotate(current_direction, degrees):
    current = directions.index(current_direction)
    step = degrees // 90
    return directions[(current + step) % 4]


def move(pos, facing, instr):
    if instr[0] in pos:
        pos[instr[0]] += instr[1]
    elif instr[0] == 'F':
        pos[facing] += instr[1]
    elif instr[0] == 'R':
        facing = rotate(facing, instr[1])
    elif instr[0] == 'L':
        facing = rotate(facing, -instr[1])

    return facing

def manhattan_distance(position):
    return abs(position['N'] - position['S']) + abs(position['E'] - position['W'])

def rotate_waypoint(wp, degrees):
    old_wp = wp.copy()
    for direction in old_wp:
        wp[direction] = old_wp[rotate(direction, degrees)]

def move_waypoint(pos, wp, instr):
    if instr[0] in wp:
        wp[instr[0]] += instr[1]
    elif instr[0] == 'F':
        for p in wp.items():
            pos[p[0]] += p[1] * instr[1]
    elif instr[0] == 'R':
        rotate_waypoint(wp, -instr[1])
    elif instr[0] == 'L':
        rotate_waypoint(wp, instr[1])


class Day12Test(unittest.TestCase):
    def test_rotate(self):
        self.assertEqual('N', rotate('W', 90))
        self.assertEqual('S', rotate('N', 180))
        self.assertEqual('E', rotate('S', -90))
        self.assertEqual('W', rotate('S', -270))

    instructions = [(instr[0], int(instr[1:])) for instr in ["F10", "N3", "F7", "R90", "F11"]]

    def test_move(self):
        pos = dict(zip(directions, [0] * 4))
        facing = 'E'
        for i in self.instructions:
            facing = move(pos, facing, i)
        self.assertEqual(25, manhattan_distance(pos))

    def test_move_wp(self):
        pos = dict(zip(directions, [0] * 4))
        wp = {'W': 0, 'E': 10, 'S': 0, 'N': 1}
        for i in self.instructions:
            move_waypoint(pos, wp, i)
        self.assertEqual(286, manhattan_distance(pos))

    def test_rotate_wp(self):
        pos = dict(zip(directions, [0] * 4))

        wp = {'N': 1, 'E': 2, 'S': 3, 'W': 4}
        move_waypoint(pos, wp, ('R', 90))
        self.assertEqual({'N': 4, 'E': 1, 'S': 2, 'W': 3}, wp)

        wp = {'N': 1, 'E': 2, 'S': 3, 'W': 4}
        move_waypoint(pos, wp, ('L', 270))
        self.assertEqual({'N': 4, 'E': 1, 'S': 2, 'W': 3}, wp)

def day12():
    instructions = [(l[0], int(l[1:])) for l in read_lines("day12input.txt")]

    start_time = time.time()

    pos = dict(zip(directions, [0] * 4))
    facing = 'E'

    for instr in instructions:
        facing = move(pos, facing, instr)
    task1 = manhattan_distance(pos)

    pos = dict(zip(directions, [0] * 4))
    wp = {'W': 0, 'E': 10, 'S': 0, 'N': 1}

    for instr in instructions:
        move_waypoint(pos, wp, instr)
    task2 = manhattan_distance(pos)

    return time.time() - start_time, task1, task2

# Day 13 - bus schedules


def get_sched(input):
    timestamp = [int(input[0])]
    buses = input[1].split(',')
    for b in buses:
        timestamp.append(0 if b == 'x' else int(b))
    return timestamp

def day13():
    sched = get_sched(read_lines("day13input.txt"))

#    sched = [939, 7, 13, 'x', 'x', 59, 'x', 31, 19]
#    sched = [i for i in sched if isinstance(i, int)]

    start_time = time.time()

    arr_time = sched[0]
    dep_time = 0
    bus_id = 0

    while not dep_time:
        for bus in sched[1:]:
            if bus and arr_time % bus == 0:
                dep_time = arr_time
                bus_id = bus
                break
        arr_time += 1

    task1 = (dep_time - sched[0]) * bus_id

    contest = {}
    i = 0
    for bus in sched[1:]:
        if bus:
            contest[bus] = bus - i
        i += 1

    # b = iter(contest)
    # print(gcd(next(b), next(b)))
    # print(gcd(next(b), next(b)))
    # print(gcd(next(b), next(b)))



    #    sched = [939, 7, 13, 'x', 'x', 59, 'x', 31, 19]

    # t % 939 = 0
    # t % 7  = 6
    # t % 13 = 11
    # t % 59 = 54
    # t % 31 = 24
    # t % 19 = 11


    # timestamp = contest[0]
    # found = False
    # while not found:
    #     print(timestamp)
    #     found = True
    #     for i in contest:
    #         if (timestamp + i) % contest[i] > 0:
    #             print(str(i) + ' ' + str(contest[i]))
    #             timestamp += contest[0]
    #             found = False
    #             break


    task2 = None

    return time.time() - start_time, task1, task2

# Day 14 bitmasks

def read_mask(input):
    mask0 = ''.join(['0' if i == '0' else '1' for i in input])
    mask1 = ''.join(['1' if i == '1' else '0' for i in input])

    return int('0b' + mask0, 2), int('0b' + mask1, 2)

def run_instructions(instructions):
    mem = dict()

    for instr in instructions:
        if instr[0] == 'mask ':
            mask0, mask1 = read_mask(instr[1].strip())
        else:
            val = int(instr[1])
            mem[int(instr[0])] = (val | mask1) & mask0

    return mem

def read_mask_v2(input):
    mask0s = [['1'] * 36]
    mask1 = ['1' if i == '1' else '0' for i in input]

    for i in range(0, 36):
        if input[i] == 'X':
            mask1[i] = '1'
            m0s = [m.copy() for m in mask0s]
            for m0 in m0s:
                m0[i] = '0'
            mask0s += m0s
    return [int('0b' + ''.join(m), 2) for m in mask0s], int('0b' + ''.join(mask1), 2)


def run_instructions_v2(instructions):
    mem = dict()

    for instr in instructions:
        if instr[0] == 'mask ':
            mask0s, mask1 = read_mask_v2(instr[1].strip())
        else:
            addr = int(instr[0])
            val = int(instr[1])
            for m0 in mask0s:
                mem[(addr | mask1) & m0] = val
    return mem


class Day13Test(unittest.TestCase):
    def test_mask(self):
        mask0, mask1 = read_mask('X' * 36)
        self.assertEqual(int('1' * 36, 2), mask0)
        self.assertEqual(0, mask1)

        mask0, mask1 = read_mask(('X' * 34) + '01')
        exp0 = int('1' * 34 + '01', 2)
        self.assertEqual(exp0, mask0)
        self.assertEqual(1, mask1)

    def test(self):
        instr = [
            "mask = XXXXXXXXXXXXXXXXXXXXXXXXXXXXX1XXXX0X",
            "mem[8] = 11",
            "mem[7] = 101",
            "mem[8] = 0",
            "mem[1] = 3"]
        instructions = [instr.split('=') if instr.startswith('mask') else re.findall('\d+', instr) for instr in instr]
        mem = run_instructions(instructions)

        self.assertEqual(3, len(mem))
        self.assertEqual(101, mem[7])
        self.assertEqual(64, mem[8])
        self.assertEqual(65, mem[1])

    def test_v2(self):
        instr = [
            "mask = 000000000000000000000000000000X1001X",
            "mem[42] = 100",
            "mask = 00000000000000000000000000000000X0XX",
            "mem[26] = 1"]
        instructions = [instr.split('=') if instr.startswith('mask') else re.findall('\d+', instr) for instr in instr]
        m0s, m1 = read_mask_v2(instructions[0][1].strip())
        self.assertEqual(59, (42 | m1) & m0s[0])
        self.assertEqual(27, (42 | m1) & m0s[1])
        self.assertEqual(58, (42 | m1) & m0s[2])
        self.assertEqual(26, (42 | m1) & m0s[3])

        m0s, m1 = read_mask_v2(instructions[2][1].strip())
        self.assertEqual(8, len(m0s))
        self.assertEqual(27, (26 | m1) & m0s[0])
        self.assertEqual(19, (26 | m1) & m0s[1])
        self.assertEqual(25, (26 | m1) & m0s[2])
        self.assertEqual(17, (26 | m1) & m0s[3])
        self.assertEqual(26, (26 | m1) & m0s[4])
        self.assertEqual(18, (26 | m1) & m0s[5])
        self.assertEqual(24, (26 | m1) & m0s[6])
        self.assertEqual(16, (26 | m1) & m0s[7])

def day14():
    instructions = [instr.split('=') if instr.startswith('mask') else re.findall('\d+', instr) for instr in read_lines("day14input.txt")]

    start_time = time.time()

    mem = run_instructions(instructions)
    task1 = sum(mem.values())

    mem = run_instructions_v2(instructions)
    task2 = sum(mem.values())

    return time.time() - start_time, task1, task2

# Day 15 memory game

# needs a 0 in the starting sequence
def number_game(input, turns):
    nums = {}

    for i in range(0, len(input)):
        nums[input[i]] = (i + 1, None)

    last_spoken = input[-1]

    for turn in range(len(input) + 1, turns + 1):
        if not nums[last_spoken][1]:
            nums[0] = (turn, nums[0][0])
            last_spoken = 0
        else:
            diff = nums[last_spoken][0] - nums[last_spoken][1]
            if diff not in nums:
                nums[diff] = (turn, None)
            else:
                nums[diff] = (turn, nums[diff][0])
            last_spoken = diff

    return last_spoken


class Day15Test(unittest.TestCase):
    def test(self):
        self.assertEqual(0, number_game([0, 3, 6], 4))
        self.assertEqual(436, number_game([0, 3, 6], 2020))


def day15():
    input = [1, 20, 8, 12, 0, 14]

    start_time = time.time()

    task1 = number_game(input, 2020)
    task2 = number_game(input, 30000000)

    return time.time() - start_time, task1, task2


# Day 18: expression tree!

class Expr:
    def __init__(self, l, op, r):
        self.left = l
        self.op = op
        self.right = r
        print(str(l), str(op), str(r))


    def eval(self):
        if not isinstance(self.left, int):
            self.left = self.left.eval()
        if not isinstance(self.right, int):
            self.right = self.right.eval()

        print(str(self.left), str(self.op), str(self.right))

        return self.op([self.left, self.right])


ops = {'*': prod, '+': sum}

# ['1', '*', '2', '+', '3']
def parse_expr(l):
    return Expr(int(l[0]), ops[l[1]], parse_expr(l[2:])) if len(l) > 1 else int(l[0])

def parse_expr2(s):
    expr = ""
    count = 0
    if s[0] == '(':
        pass



class Test18(unittest.TestCase):
    input1 = "1 + 2 * 3 + 4 * 5 + 6"
    input2 = "1 + (2 * 3) + (4 * (5 + 6))"

    input1 = "(((((1 + 2) * 3) + 4) * 5) + 6)"
    input2 = "((1 + (2 * 3)) + (4 * (5 + 6)))"

    #print(list(reversed(input2)))
    #print(input2)

    def test_parse(self):
        input1 = self.input1.split()
        input1.reverse()
#        self.assertEqual(71, parse_expr(input1).eval())

    def test_parse_brackets(self):
 #       parse_expr2(")2 + 1(")
        pass


# Day 19: Decode scrambled messages

def parse_one_rule(rule, rules):
    if (isinstance(rule, int)):
        return rules[rule]
    return rule

def parse_pair_rule(rule, rules):
    r = [parse_one_rule(rule[0], rules), parse_one_rule(rule[1], rules)]

    for i in r:
        if not isinstance(i, str):
            return r
    return ''.join(r)


def parse_rule(rule, rules):
    rule0 = parse_pair_rule(rule, rules)
    #print(rule0)

    while not isinstance(rule0[0], str):
        if isinstance(rule0[0], int):
            rule0[0] = parse_one_rule(rule0[0], rules)
        else:
            rule0[0] = parse_pair_rule(rule0[0], rules)

    for rule in rule0[1]:
        #print(rule)
        for r in rule:
            r = parse_pair_rule(r, rules)
            # while not isinstance(r[0], str):
            #     if isinstance(r[0], int):
            #         r[0] = parse_one_rule(r[0], rules)
            #     else:
            #         r[0] = parse_pair_rule(r[0], rules)

#    rule0[1][0] = parse_pair_rule(rule0[1][0], rules)
#    rule0[1][1] = parse_pair_rule(rule0[1][1], rules)

    #print(rule0)


class Test19(unittest.TestCase):
    input = ["1 2", "a", ["1 3", "3 1"], "b"]
    input2 = {0: [1, 2], 1: "a", 2: [[1, 3], [3, 1]], 3: "b"}
    input3 = [[[1, 2]], "a", [[1, 3], [3, 1]], "b"]

    def test(self):
        pass
#        rule0 = parse_rule(self.input2[0], self.input2)
#        self.assertEqual({"aab", "aba"}, rule0)

#        self.assertEqual({"aaaabb", "aaabab", "abbabb", "abbbab", "aabaab", "aabbbb", "abaaab", "ababbb"}, rules)

# Day 20

def reverse_str(s):
    return ''.join(list(reversed(s)))

def get_square_outline(l):
    id = int(re.search(r"(\d+)", l[0]).groups()[0])
    right = ''.join([i[-1] for i in l[1:11]])
    left = ''.join([i[0] for i in l[1:11]])

    return (id,
            [l[1], reverse_str(l[1]),
             right, reverse_str(right),
             l[10], reverse_str(l[10]),
             left, reverse_str(left)])

def get_outlines(squares):
    outlines = {}
    for i in range(0, len(squares)//12):
        key, val = get_square_outline(squares[i * 12: (i + 1) * 12])
        outlines[key] = val
    return outlines

def find_corners(image_map):
    corners = []
    for c in image_map:
        if len(image_map[c]) == 2:
            corners.append(c)
    return corners

def build_image_map(outlines):
    image_map = {k: [] for k in outlines.keys()}

    for i in outlines:
        for j in outlines:
            if i < j:
                intersect = set(outlines[i]).intersection(set(outlines[j]))
                if len(intersect):
                    image_map[i].append(j)
                    image_map[j].append(i)
    return image_map

def build_image(corners, image_map):
    # select one corner as anchor
    c0 = corners[0]
    image = [
        [c0, image_map[c0][0]],
        [image_map[c0][1]]]

    # build image from that corner

    # lower right corner
    common = set(image_map[image[-1][-1]]).intersection(set(image_map[image[-2][-1]]))
    common.remove(c0)
    image[-1].append(common.pop())

    # next right
    right = set(image_map[image[-2][-1]]).difference(set([image[-2][-2], image[-1][-1]]))
    image[-2].append(right.pop())

    # lower right corner
    common = set(image_map[image[-1][-1]]).intersection(set(image_map[image[-2][-1]]))
    common.remove(image[-2][-2])
    image[-1].append(common.pop())

    # start next row
    down = set(image_map[image[-1][0]]).difference(set([image[-2][0], image[-1][1]]))
    image.append([down.pop()])

    # lower right corner
    common = set(image_map[image[-1][0]]).intersection(set(image_map[image[-2][1]]))
    common.remove(image[-2][0])
    image[-1].append(common.pop())

    # next right
    right = set(image_map[image[-2][-1]]).difference(set([image[-2][-2], image[-1][-1]]))
    image[-1].append(right.pop())

    return image

edge_index = ["top", "flip top", "right", "flip right", "bottom", "flip bottom", "left", "flip left"]

def get_right_flip_rotation(index):
    if edge_index[index] == "left":
        flip_rotation = (0, 0)
    elif edge_index[index] == "flip left":
        flip_rotation = ("top-bottom", 0)
    elif edge_index[index] == "right":
        flip_rotation = ("left-right", 0)
    elif edge_index[index] == "flip right":
        flip_rotation = (["left-right", "top-bottom"], 0)
    elif edge_index[index] == "flip top":
        flip_rotation = (0, -90)
    elif edge_index[index] == "top":
        flip_rotation = ("left-right", -90)
    elif edge_index[index] == "bottom":
        flip_rotation = (0, 90)
    elif edge_index[index] == "flip bottom":
        flip_rotation = ("left-right", 90)

    return flip_rotation


def get_bottom_flip_rotation(index):
    if edge_index[index] == "top":
        flip_rotation = (0, 0)
    elif edge_index[index] == "flip top":
        flip_rotation = ("left-right", 0)
    elif edge_index[index] == "bottom":
        flip_rotation = ("top-bottom", 0)
    elif edge_index[index] == "flip bottom":
        flip_rotation = (["left-right", "top-bottom"], 0)
    elif edge_index[index] == "flip left":
        flip_rotation = (0, 90)
    elif edge_index[index] == "left":
        flip_rotation = ("top-bottom", 90)
    elif edge_index[index] == "right":
        flip_rotation = (0, -90)
    elif edge_index[index] == "flip right":
        flip_rotation = ("top-bottom", -90)

    return flip_rotation

# ["top", "flip top", "right", "flip right", "bottom", "flip bottom", "left", "flip left"]

index_edge = dict(zip(edge_index, range(0, len(edge_index))))

def flip_rotate(image, f_r):
    im = {}
    for f in f_r[0]:
        if f == "top-bottom":
            im["top"] = image[index_edge["bottom"]]
            im["flip top"] = image[index_edge["flip bottom"]]
            im["bottom"] = image[index_edge["top"]]
            im["flip bottom"] = image[index_edge["flip top"]]
            im["right"] = image[index_edge["flip right"]]
            im["flip right"] = image[index_edge["right"]]
            im["left"] = image[index_edge["flip left"]]
            im["flip left"] = image[index_edge["left"]]
        else:
            im["top"] = image[index_edge["flip top"]]
            im["flip top"] = image[index_edge["top"]]
            im["bottom"] = image[index_edge["flip bottom"]]
            im["flip bottom"] = image[index_edge["bottom"]]
            im["right"] = image[index_edge["left"]]
            im["flip right"] = image[index_edge["flip left"]]
            im["left"] = image[index_edge["right"]]
            im["flip left"] = image[index_edge["flip right"]]

    if f_r[1] == 90:
        im["top"] = image[index_edge["flip left"]]
        im["flip top"] = image[index_edge["left"]]
        im["bottom"] = image[index_edge["flip right"]]
        im["flip bottom"] = image[index_edge["right"]]
        im["right"] = image[index_edge["top"]]
        im["flip right"] = image[index_edge["flip top"]]
        im["left"] = image[index_edge["bottom"]]
        im["flip left"] = image[index_edge["flip bottom"]]
    else:
        im["top"] = image[index_edge["right"]]
        im["flip top"] = image[index_edge["flip right"]]
        im["bottom"] = image[index_edge["left"]]
        im["flip bottom"] = image[index_edge["flip left"]]
        im["right"] = image[index_edge["flip bottom"]]
        im["flip right"] = image[index_edge["bottom"]]
        im["left"] = image[index_edge["flip top"]]
        im["flip left"] = image[index_edge["top"]]

    return im


def get_flip_rotate_origin(right_edge, bottom_edge):
    flip = 0
    rot = 0

    if right_edge == "right":
        if bottom_edge == "top":
            flip = "top-bottom"
    elif right_edge == "left":
        flip = "left-right"
        if bottom_edge == "top":
            flip = ["left-right", "top-bottom"]
    elif right_edge == "top":
        rot = 90
        if bottom_edge == "left":
            flip = "left-right"
    else:
        rot = -90
        if bottom_edge == "right":
            flip = "left-right"

    return (flip, rot)


class Test20(unittest.TestCase):
    inp = [
        "Tile 2311:",
        "..##.#..#.",
        "##..#.....",
        "#...##..#.",
        "####.#...#",
        "##.##.###.",
        "##...#.###",
        ".#.#.#..##",
        "..#....#..",
        "###...#.#.",
        "..###..###",
        "",
        "Tile 1951:",
        "#.##...##.",
        "#.####...#",
        ".....#..##",
        "#...######",
        ".##.#....#",
        ".###.#####",
        "###.##.##.",
        ".###....#.",
        "..#.#..#.#",
        "#...##.#..",
        "",
        "Tile 1171:",
        "####...##.",
        "#..##.#..#",
        "##.#..#.#.",
        ".###.####.",
        "..###.####",
        ".##....##.",
        ".#...####.",
        "#.##.####.",
        "####..#...",
        ".....##...",
        "",
        "Tile 1427:",
        "###.##.#..",
        ".#..#.##..",
        ".#.##.#..#",
        "#.#.#.##.#",
        "....#...##",
        "...##..##.",
        "...#.#####",
        ".#.####.#.",
        "..#..###.#",
        "..##.#..#.",
        "",
        "Tile 1489:",
        "##.#.#....",
        "..##...#..",
        ".##..##...",
        "..#...#...",
        "#####...#.",
        "#..#.#.#.#",
        "...#.#.#..",
        "##.#...##.",
        "..##.##.##",
        "###.##.#..",
        "",
        "Tile 2473:",
        "#....####.",
        "#..#.##...",
        "#.##..#...",
        "######.#.#",
        ".#...#.#.#",
        ".#########",
        ".###.#..#.",
        "########.#",
        "##...##.#.",
        "..###.#.#.",
        "",
        "Tile 2971:",
        "..#.#....#",
        "#...###...",
        "#.#.###...",
        "##.##..#..",
        ".#####..##",
        ".#..####.#",
        "#..#.#..#.",
        "..####.###",
        "..#.#.###.",
        "...#.#.#.#",
        "",
        "Tile 2729:",
        "...#.#.#.#",
        "####.#....",
        "..#.#.....",
        "....#..#.#",
        ".##..##.#.",
        ".#.####...",
        "####.#.#..",
        "##.####...",
        "##..#.##..",
        "#.##...##.",
        "",
        "Tile 3079:",
        "#.#.#####.",
        ".#..######",
        "..#.......",
        "######....",
        "####.#..#.",
        ".#...#.##.",
        "#.#####.##",
        "..#.###...",
        "..#.......",
        "..#.###...",
        " "]


    def test(self):
        outlines = get_outlines(self.inp)

        exp_outlines = dict()
        for i in range(0, 9):
            key, val = get_square_outline(self.inp[i*12: (i+1)*12])
            val += [''.join(list(reversed(s))) for s in val]
            exp_outlines[key] = val

        self.assertEqual(len(exp_outlines), len(outlines))
        self.assertEqual(set(exp_outlines), set(outlines))

        image_map = build_image_map(outlines)

        corners = find_corners(image_map)
        self.assertEqual(20899048083289, prod(corners))

        image = build_image(corners, image_map)

        # flip-rotate corner 0 to match neighbours
        right = set(outlines[image[0][0]]).intersection(set(outlines[image[0][1]]))
        right_index = outlines[image[0][0]].index(right.pop())
        # if it's a flip edge, choose the original orientation instead
        right_index -= right_index % 2
        right_edge = edge_index[right_index]

        bottom = set(outlines[image[0][0]]).intersection(set(outlines[image[1][0]]))
        bottom_index = outlines[image[0][0]].index(bottom.pop())
        bottom_index -= bottom_index % 2
        bottom_edge = edge_index[bottom_index]

        c0_f_r = get_flip_rotate_origin(right_edge, bottom_edge)
        self.assertEqual(("top-bottom", 0), c0_f_r)

        image_rotation = {image[0][0]: c0_f_r}


        # get the flip-rotated right edge of previous image, then flip-rotate the one to the right

        i = 0

        for j in range(1, len(image[0])):
            right = outlines[image[i][j-1]][2]
            index = outlines[image[i][j]].index(right)
            image_rotation[image[i][j]] = get_right_flip_rotation(index)

        print(image)
        print(image_rotation)

        # continue with row below

        i = 1
        j = 0

        bottom = outlines[image[i-1][j]][0]
        print(bottom)
        print(image[i][j])
        print(outlines[image[i][j]])

        index = outlines[image[i][j]].index(bottom)
        image_rotation[image[i][j]] = get_bottom_flip_rotation(index)

        print(image_rotation)


def day20():
    inp = [s.strip() for s in read_lines("day20input.txt")]

    outlines = get_outlines(inp)
    image_map = build_image_map(outlines)
    corners = find_corners(image_map)


    #print(image_map)




    return 0, prod(corners), 0


# Day 22 - space cards!

def play_round(hand1, hand2):
    card1 = hand1.popleft()
    card2 = hand2.popleft()

    if card1 > card2:
        hand1.append(card1)
        hand1.append(card2)
    else:
        hand2.append(card2)
        hand2.append(card1)

def check_recurring_deck(deck, history):
    return deck in history

def play_recurring_deck(hand1, hand2, history):
    if hand1 in history[0]:
        print("hand1 recurred")
        return 1
    if hand2 in history[1]:
        print("hand2 recurred")
        return 1

    history[0].append(hand1.copy())
    history[1].append(hand2.copy())

    card1 = hand1.popleft()
    card2 = hand2.popleft()

    if len(hand1) >= card1 and len(hand2) >= card2:
        sub_winner = play_recursive_combat(deque(list(hand1)[:card1]), deque(list(hand2)[:card2]))
    else:
        sub_winner = 1 if card1 > card2 else 2

    if sub_winner == 1:
        hand1.append(card1)
        hand1.append(card2)
    else:
        hand2.append(card2)
        hand2.append(card1)

    if not len(hand1):
        return 2

    if not len(hand2):
        return 1

    return False

def play_recursive_combat(hand1, hand2):
    history = [[], []]
    winner = play_recurring_deck(hand1, hand2, history)

    while not winner:
        winner = play_recurring_deck(hand1, hand2, history)

    return winner

class Test2(unittest.TestCase):
    hand1 = deque([9, 2, 6, 3, 1])
    hand2 = deque([5, 8, 4, 7, 10])

    def test(self):
        hand1 = self.hand1.copy()
        hand2 = self.hand2.copy()
        play_round(hand1, hand2)
        self.assertEqual(6, len(hand1))
        self.assertEqual(4, len(hand2))

        while len(hand1) and len(hand2):
            winner = play_round(hand1, hand2)

        win_hand = hand1 if winner == 1 else hand2
        self.assertEqual(306, sum([a * b for a, b in zip(win_hand, range(len(win_hand), 0, -1))]))

    # should trigger recursion break rule
    def test_recursion(self):
        hand1 = deque([43, 19])
        hand2 = deque([2, 29, 14])

        winner = play_recursive_combat(hand1, hand2)
        self.assertEqual(1, winner)

    def test_history(self):
        history = [[]]
        self.assertFalse(self.hand1 in history[0])
        history[0].append(deque([3, 4]))
        history[0].append(self.hand1.copy())
        history[0].append(deque([1,4,6]))
        self.assertTrue(self.hand1 in history[0])
        self.assertFalse(self.hand2 in history[0])

        self.assertEqual(1, play_recurring_deck(deque([1, 2, 3]), deque([4,5,6]), [[deque([1, 2, 3])], []]))
        self.assertEqual(1, play_recurring_deck(deque([1, 2, 3]), deque([4,5,6]), [[], [deque([4,5,6])]]))


    def test_recursive_combat(self):
        self.assertEqual(deque([1, 2, 3]), deque([1, 2, 3]))

        hand1 = self.hand1.copy()
        hand2 = self.hand2.copy()

        history1 = []
        history2 = []

        self.assertFalse(play_recurring_deck(hand1, hand2, [history1, history2]))
        self.assertEqual(deque([2, 6, 3, 1, 9, 5]), hand1)
        self.assertEqual(deque([8, 4, 7, 10]), hand2)
        self.assertTrue(self.hand1 in history1)
        self.assertTrue(self.hand2 in history2)

        self.assertFalse(play_recurring_deck(hand1, hand2, [history1, history2]))
        self.assertEqual(deque([6, 3, 1, 9, 5]), hand1)
        self.assertEqual(deque([4, 7, 10, 8, 2]), hand2)

        self.assertFalse(play_recurring_deck(hand1, hand2, [history1, history2]))
        self.assertEqual(deque([3, 1, 9, 5, 6, 4]), hand1)
        self.assertEqual(deque([7, 10, 8, 2]), hand2)

        self.assertFalse(play_recurring_deck(hand1, hand2, [history1, history2]))
        self.assertEqual(deque([1, 9, 5, 6, 4]), hand1)
        self.assertEqual(deque([10, 8, 2, 7, 3]), hand2)

        self.assertFalse(play_recurring_deck(hand1, hand2, [history1, history2]))
        self.assertEqual(deque([9, 5, 6, 4]), hand1)
        self.assertEqual(deque([8, 2, 7, 3, 10, 1]), hand2)

        self.assertFalse(play_recurring_deck(hand1, hand2, [history1, history2]))
        self.assertEqual(deque([5, 6, 4, 9, 8]), hand1)
        self.assertEqual(deque([2, 7, 3, 10, 1]), hand2)

        self.assertFalse(play_recurring_deck(hand1, hand2, [history1, history2]))
        self.assertEqual(deque([6, 4, 9, 8, 5, 2]), hand1)
        self.assertEqual(deque([7, 3, 10, 1]), hand2)

        self.assertFalse(play_recurring_deck(hand1, hand2, [history1, history2]))
        self.assertEqual(deque([4, 9, 8, 5, 2]), hand1)
        self.assertEqual(deque([3, 10, 1, 7, 6]), hand2)

        self.assertFalse(play_recurring_deck(hand1, hand2, [history1, history2]))
        self.assertEqual(deque([9, 8, 5, 2]), hand1)
        self.assertEqual(deque([10, 1, 7, 6, 3, 4]), hand2)

        self.assertFalse(play_recurring_deck(hand1, hand2, [history1, history2]))
        self.assertEqual(deque([8, 5, 2]), hand1)
        self.assertEqual(deque([1, 7, 6, 3, 4, 10, 9]), hand2)

        self.assertFalse(play_recurring_deck(hand1, hand2, [history1, history2]))
        self.assertEqual(deque([5, 2, 8, 1]), hand1)
        self.assertEqual(deque([7, 6, 3, 4, 10, 9]), hand2)

        self.assertFalse(play_recurring_deck(hand1, hand2, [history1, history2]))
        self.assertEqual(deque([2, 8, 1]), hand1)
        self.assertEqual(deque([6, 3, 4, 10, 9, 7, 5]), hand2)

        self.assertFalse(play_recurring_deck(hand1, hand2, [history1, history2]))
        self.assertEqual(deque([8, 1]), hand1)
        self.assertEqual(deque([3, 4, 10, 9, 7, 5, 6, 2]), hand2)

        self.assertFalse(play_recurring_deck(hand1, hand2, [history1, history2]))
        self.assertEqual(deque([1, 8, 3]), hand1)
        self.assertEqual(deque([4, 10, 9, 7, 5, 6, 2]), hand2)

        self.assertFalse(play_recurring_deck(hand1, hand2, [history1, history2]))
        self.assertEqual(deque([8, 3]), hand1)
        self.assertEqual(deque([10, 9, 7, 5, 6, 2, 4, 1]), hand2)

        self.assertFalse(play_recurring_deck(hand1, hand2, [history1, history2]))
        self.assertEqual(deque([3]), hand1)
        self.assertEqual(deque([9, 7, 5, 6, 2, 4, 1, 10, 8]), hand2)

        self.assertEqual(2, play_recurring_deck(hand1, hand2, [history1, history2]))
        self.assertEqual(deque([]), hand1)
        self.assertEqual(deque([7, 5, 6, 2, 4, 1, 10, 8, 9, 3]), hand2)

        hand1 = self.hand1.copy()
        hand2 = self.hand2.copy()
        self.assertEqual(2, play_recursive_combat(hand1, hand2))
        self.assertEqual(291, sum([a * b for a, b in zip(list(hand2), range(len(hand2), 0, -1))]))


def day22():
    inp = read_lines('day22input.txt')
    deck1 = deque([int(line) for line in inp[1:26]])
    deck2 = deque([int(line) for line in inp[28:]])

    hand1 = deck1.copy()
    hand2 = deck2.copy()

    while len(hand1) and len(hand2):
        play_round(hand1, hand2)

    winner = hand1 if len(hand1) else hand2
    task1 = sum([a * b for a, b in zip(winner, range(len(winner), 0, -1))])

    hand1 = deck1.copy()
    hand2 = deck2.copy()

    winner = play_recursive_combat(hand1, hand2)
    winner = hand2 if winner == 2 else hand1
    task2 = sum([a * b for a, b in zip(winner, range(len(winner), 0, -1))])

    return 0, task1, task2

# Main


def run(day):
    run_time, task1, task2 = day()
    print(day.__name__ + ": %.6s s - " % run_time + str(task1) + " " + str(task2))

def run_tests():
    test_files = glob.glob('*.py')
    module_strings = [test_file[0:len(test_file) - 3] for test_file in test_files]
    suites = [unittest.defaultTestLoader.loadTestsFromName(test_file) for test_file in module_strings]
    test_suite = unittest.TestSuite(suites)
    unittest.TextTestRunner().run(test_suite)


if __name__ == '__main__':
    run_tests()
    for i in range(22, 23):
        run(eval("day" + str(i)))

