# coding=utf-8

from itertools import combinations
from itertools import permutations
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

# Day 11 - game of life

seat_dict = {'#': 2, 'L': 1, '.': 0}

def get_matrix(lines):
    return [[seat_dict[ch] for ch in l] for l in lines]

def occupied_neighbours(r, c, seats):
    no_rows = len(seats)
    no_cols = len(seats[0])
    count = 0

    max_no = 4 if seats[r][c] == 2 else 0

    for i in range(r - 1, r + 2):
        if i < 0:
            continue
        if i >= no_rows:
            break
        if count > max_no:
            break

        for j in range(c - 1, c + 2):
            if j >= 0 and j < no_cols:
                count += seats[i][j] == 2

    return count - (seats[r][c] == 2)

def left_neighbour(r, c, seats):
    for x in reversed(seats[r][0:c]):
        if x > 0:
            return x == 2
    return 0

def right_neighbour(r, c, seats):
    for x in seats[r][c+1:]:
        if x > 0:
            return x == 2
    return 0

def up_neighbour(r, c, seats):
    for row in reversed(seats[0:r]):
        if row[c] > 0:
            return row[c] == 2
    return 0

def down_neighbour(r, c, seats):
    for row in seats[r+1:]:
        if row[c] > 0:
            return row[c] == 2
    return 0

def left_up_neighbour(r, c, seats):
    r -= 1
    c -= 1
    while r >= 0 and c >= 0:
        if seats[r][c]:
            return seats[r][c] == 2
        r -= 1
        c -= 1

    return 0

def left_down_neighbour(row, col, seats):
    r = row + 1
    c = col - 1
    while r < len(seats) and c >= 0:
        if seats[r][c]:
            return seats[r][c] == 2
        r += 1
        c -= 1

    return 0

def right_up_neighbour(r, c, seats):
    r -= 1
    c += 1
    while r >= 0 and c < len(seats[0]):
        if seats[r][c]:
            return seats[r][c] == 2
        r -= 1
        c += 1

    return 0


def right_down_neighbour(r, c, seats):
    r += 1
    c += 1
    while r < len(seats) and c < len(seats[0]):
        if seats[r][c]:
            return seats[r][c] == 2
        r += 1
        c += 1

    return 0

neighbour_counters = [
    left_neighbour, right_neighbour, up_neighbour, down_neighbour,
    left_up_neighbour, left_down_neighbour, right_up_neighbour, right_down_neighbour]


def seen_neighbours(r, c, seats):
    count = 0
    max_no = 5 if seats[r][c] == 2 else 0

    for counter in neighbour_counters:
        if count > max_no:
            break
        count += counter(r, c, seats)
    return count


def move_people(seats, max_n, neighbour_alg):
    new_seats = [row.copy() for row in seats]

    for r in range(0, len(seats)):
        for c in range(0, len(seats[0])):
            if seats[r][c]:
                neighbours = neighbour_alg(r, c, seats)
                if seats[r][c] == 1 and not neighbours:
                    new_seats[r][c] = 2
                elif seats[r][c] == 2 and neighbours >= max_n:
                    new_seats[r][c] = 1

    return new_seats


class Day11Test(unittest.TestCase):
    seats = [
        "L.LL.LL.LL",
        "LLLLLLL.LL",
        "L.L.L..L..",
        "LLLL.LL.LL",
        "L.LL.LL.LL",
        "L.LLLLL.LL",
        "..L.L.....",
        "LLLLLLLLLL",
        "L.LLLLLL.L",
        "L.LLLLL.LL"]

    exp = [
        "#.#L.L#.##",
        "#LLL#LL.L#",
        "L.#.L..#..",
        "#L##.##.L#",
        "#.#L.LL.LL",
        "#.#L#L#.##",
        "..L.L.....",
        "#L#L##L#L#",
        "#.LLLLLL.L",
        "#.#L#L#.##"]

    seats2 = [
        "#.L#.L#.L#",
        "#LLLLLL.LL",
        "L.L.L..#..",
        "##L#.#L.L#",
        "L.L#.LL.L#",
        "#.LLLL#.LL",
        "..#.L.....",
        "LLL###LLL#",
        "#.LLLLL#.L",
        "#.L#LL#.L#"]

    def test(self):
        seats = get_matrix(self.seats)
        new_seats = move_people(seats, 4, occupied_neighbours)
        while seats != new_seats:
            seats = [i.copy() for i in new_seats]
            new_seats = move_people(seats, 4, occupied_neighbours)

        exp = get_matrix(self.exp)
        self.assertEqual(exp, new_seats)

    def test_left_neighbour(self):
        seats = get_matrix(self.exp)
        self.assertTrue(left_neighbour(0, 3, seats))
        self.assertFalse(left_neighbour(0, 5, seats))
        self.assertFalse(left_neighbour(0, 0, seats))

    def test_seen_neighbours(self):
        seats = get_matrix(self.exp)
        self.assertEqual(1, seen_neighbours(0, 3, seats))
        self.assertEqual(6, seen_neighbours(3, 2, seats))

def count_occupied(seats):
    count = 0
    for row in seats:
        count += sum(i == 2 for i in row)
    return count

def day11():
    seats = [l.strip() for l in read_lines("day11input.txt")]
    seats = get_matrix(seats)

    start_time = time.time()

    new_seats = move_people(seats, 4, occupied_neighbours)
    prev_seats = seats
    count_iter = 0

    while prev_seats != new_seats:
        count_iter += 1
        prev_seats = new_seats
        new_seats = move_people(prev_seats, 4, occupied_neighbours)

#    print(count_iter)

    task1 = count_occupied(new_seats)

    run_time = time.time() - start_time

 #   start_time = time.time()

    new_seats = move_people(seats, 5, seen_neighbours)
#    new_seats = seats
    prev_seats = seats
    count_iter = 0
    while prev_seats != new_seats:
        count_iter += 1
        prev_seats = new_seats
        new_seats = move_people(prev_seats, 5, seen_neighbours)

    task2 = count_occupied(new_seats)

    run_time = time.time() - start_time


#    print(count_iter)

    return run_time, task1, task2

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


# Main


def run(day):
    run_time, task1, task2 = day()
    print(day.__name__ + ": %.6s s - " % run_time + str(task1) + " " + str(task2))


if __name__ == '__main__':
    for i in range(1, 13):
        run(eval("day" + str(i)))
    unittest.main()

