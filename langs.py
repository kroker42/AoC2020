# coding=utf-8

import unittest
import re
import itertools
from filereader import read_lines

# Day 16

def parse_tickets(input):
    lines = iter(input)
    line = next(lines)

    ranges = {}
    while line and not line.isspace():
        field = line.split(':')[0]
        v = re.findall(r"(\d+)-(\d+)", line)
        v = [range(int(i[0]), int(i[1]) + 1) for i in v]
        ranges[field] = v
        line = next(lines)

    next(lines)
    ticket = [int(i) for i in next(lines).split(',')]

    next(lines)
    next(lines)
    line = next(lines)

    nearby_tickets = []
    while line and not line.isspace():
        nearby_tickets.append([int(i) for i in line.split(',')])
        line = next(lines, "")

    return ranges, nearby_tickets, ticket

def is_valid(val, ranges):
    for r in ranges:
        if val in r:
            return True
    return False

def get_invalid_value(ticket, ranges):
    for i in ticket:
        val = i
        valid = is_valid(i, ranges)
        if not valid:
            return val
    return 0

def map_fields(fields, no_fields):
    field_map = {}
    while len(field_map) < no_fields:
        for col_fld in fields.items():
            if len(col_fld[1]) == 1:
                field_map[col_fld[1][0]] = col_fld[0]
            else:
                for f in col_fld[1]:
                    if f in field_map:
                        col_fld[1].remove(f)
    return field_map

def possible_fields2(tickets, ranges):
    fields = {k: [] for k in ranges}

    for r in ranges.items():
        for i in range(0, len(tickets[0])):
            for t in tickets:
                valid = is_valid(t[i], r[1])
                if not valid:
                    break
            if valid:
                fields[r[0]].append(i)
    return fields


class Day16Test(unittest.TestCase):
    tickets = [
        "class: 1-3 or 5-7",
        "row: 6-11 or 33-44",
        "seat: 13-40 or 45-50",
        "",
        "your ticket:",
        "7,1,14",
        "",
        "nearby tickets:",
        "7,3,47",
        "40,4,50",
        "55,2,20",
        "38,6,12"]

    def test(self):
        ranges, nearby_tickets, ticket = parse_tickets(self.tickets)
        self.assertEqual([range(1, 4), range(5, 8)], ranges['class'])

        # print(ranges)
        # print(nearby_tickets)
        # print(ticket)

        all_ranges = list(itertools.chain.from_iterable(ranges.values()))

#        print(all_ranges)

        self.assertEqual(0, get_invalid_value(nearby_tickets[0], all_ranges))
        self.assertEqual(4, get_invalid_value(nearby_tickets[1], all_ranges))
        self.assertEqual(55, get_invalid_value(nearby_tickets[2], all_ranges))
        self.assertEqual(12, get_invalid_value(nearby_tickets[3], all_ranges))

    def test_categories(self):
        input = [
            "class: 0-1 or 4-19",
            "row: 0-5 or 8-19",
            "seat: 0-13 or 16-19",
            "",
            "your ticket:",
            "11,12,13",
            "",
            "nearby tickets:",
            "3,9,18",
            "15,1,5",
            "5,14,9"]

        ranges, nearby_tickets, ticket = parse_tickets(input)
        fields = possible_fields2(nearby_tickets, ranges)
  #      print(fields)
        field_map = map_fields(fields, len(ranges))
   #     print(field_map)

def day16():
    input = read_lines("day16input.txt")

    ranges, nearby_tickets, ticket = parse_tickets(input)
    all_ranges = list(itertools.chain.from_iterable(ranges.values()))

    start_time = time.time()

    task1 = 0

    valid_tickets = []
    for t in nearby_tickets:
        val = get_invalid_value(t, all_ranges)
        task1 += val
        if val == 0:
            valid_tickets.append(t)

    fields = possible_fields2(valid_tickets, ranges)
    sorted_fields = list(fields.items())
    sorted_fields.sort(key=lambda a: len(a[1]))

    seen = []
    for f in sorted_fields:
        d = set(f[1]) - set(seen)
        if len(d) == 1:
            seen.append(d.pop())
            fields[f[0]] = seen[-1]
        else:
            fields[f[0]] = None

    dep_fields = ["departure location", "departure station", "departure platform",
                  "departure track", "departure date", "departure time"]

    task2 = prod([ticket[fields[f]] for f in dep_fields])

    return time.time() - start_time, task1, task2

# Day 21: Safe food


def parse_food(food):
    ingredients, allergens = re.split(r'\(', food)
    ingredients = ingredients.strip().split(' ')
    allergens = re.findall(r"\w+", allergens)[1:]
    return (ingredients, allergens)


def get_possible_allergens(foods):
    allergens = dict()
    ingredients = dict()

    for f in foods:
        for i in f[0]:
            if i in ingredients:
                ingredients[i] += 1
            else:
                ingredients[i] = 1

        ingr = set(f[0])
        for a in f[1]:
            if a in allergens:
                allergens[a] = allergens[a].intersection(ingr)
            else:
                allergens[a] = ingr

    return allergens, ingredients


class Test21(unittest.TestCase):
    food_list = [
        "mxmxvkd kfcds sqjhc nhms (contains dairy, fish)",
        "trh fvjkl sbzzf mxmxvkd (contains dairy)",
        "sqjhc fvjkl (contains soy)",
        "sqjhc mxmxvkd sbzzf (contains fish)"]

    def test(self):
        foods = [parse_food(f) for f in self.food_list]
        allergens, ingredients = get_possible_allergens(foods)
        self.assertEqual({'dairy': {'mxmxvkd'}, 'fish': {'mxmxvkd', 'sqjhc'}, 'soy': {'sqjhc', 'fvjkl'}}, allergens)
        self.assertEqual({'mxmxvkd': 3, 'kfcds': 1, 'sqjhc': 3, 'nhms': 1, 'trh': 1, 'fvjkl': 2, 'sbzzf': 2}, ingredients)

        poss_allergenic_ingredients = set()
        for i in allergens.values():
            poss_allergenic_ingredients = poss_allergenic_ingredients.union(i)
        self.assertEqual({'fvjkl', 'sqjhc', 'mxmxvkd'}, poss_allergenic_ingredients)

        safe_ingredients = set(ingredients.keys()).difference(poss_allergenic_ingredients)
        count = sum([ingredients[i] for i in safe_ingredients])
        self.assertEqual(5, count)

        allergens = {k : list(v) for (k,v) in allergens.items()}
        allergen_map = map_fields(allergens, len(poss_allergenic_ingredients))
        self.assertEqual({'mxmxvkd': 'dairy', 'sqjhc': 'fish', 'fvjkl': 'soy'}, allergen_map)


def day21():
    foods = [parse_food(l) for l in read_lines('day21input.txt')]
    allergens, ingredients = get_possible_allergens(foods)

    poss_allergenic_ingredients = set()
    for i in allergens.values():
        poss_allergenic_ingredients = poss_allergenic_ingredients.union(i)

    safe_ingredients = set(ingredients.keys()).difference(poss_allergenic_ingredients)
    task1 = sum([ingredients[i] for i in safe_ingredients])

    allergens = {k: list(v) for (k, v) in allergens.items()}
    allergen_map = map_fields(allergens, len(poss_allergenic_ingredients))

    sorted_allergens = {k: v for k, v in sorted(allergen_map.items(), key=lambda item: item[1])}
    task2 = ','.join(sorted_allergens.keys())

    return 0, task1, task2



