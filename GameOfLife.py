# coding=utf-8

from filereader import read_lines
import time
import unittest

import np

def get_matrix(lines, symbols):
    return [[symbols[ch] for ch in l] for l in lines]

# Day 11 - game of life

seat_dict = {'#': 2, 'L': 1, '.': 0}

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
        seats = get_matrix(self.seats, seat_dict)
        new_seats = move_people(seats, 4, occupied_neighbours)
        while seats != new_seats:
            seats = [i.copy() for i in new_seats]
            new_seats = move_people(seats, 4, occupied_neighbours)

        self.assertEqual(seats, new_seats)

    def test_left_neighbour(self):
        seats = get_matrix(self.exp, seat_dict)
        self.assertTrue(left_neighbour(0, 3, seats))
        self.assertFalse(left_neighbour(0, 5, seats))
        self.assertFalse(left_neighbour(0, 0, seats))

    def test_seen_neighbours(self):
        seats = get_matrix(self.exp, seat_dict)
        self.assertEqual(1, seen_neighbours(0, 3, seats))
        self.assertEqual(6, seen_neighbours(3, 2, seats))

def count_occupied(seats):
    count = 0
    for row in seats:
        count += sum(i == 2 for i in row)
    return count

def day11():
    seats = [l.strip() for l in read_lines("day11input.txt")]
    seats = get_matrix(seats, seat_dict)

    start_time = time.time()

    new_seats = move_people(seats, 4, occupied_neighbours)
    prev_seats = seats

    while prev_seats != new_seats:
        prev_seats = new_seats
        new_seats = move_people(prev_seats, 4, occupied_neighbours)

    task1 = count_occupied(new_seats)

    run_time = time.time() - start_time

 #   start_time = time.time()

    new_seats = move_people(seats, 5, seen_neighbours)
#    new_seats = seats
    prev_seats = seats

    while prev_seats != new_seats:
        prev_seats = new_seats
        new_seats = move_people(prev_seats, 5, seen_neighbours)

    task2 = count_occupied(new_seats)

    run_time = time.time() - start_time

    return run_time, task1, task2

# Day 17 - Game of Life, rev'd

life_dict = {'.': 0, '#': 1}

def build_grid(matrix, no_iter):
    grid_size = len(matrix) + 2 * no_iter
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=int)

    k = grid_size // 2 + 1
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix)):
            grid[k, i+no_iter, j+no_iter] = matrix[i][j]

    return grid, (k, no_iter, no_iter)

def check_neighbours(point, grid):
    p0, p1, p2 = point
    count = 0
    for i in range(p0 - 1, p0 + 2):
        for j in range(p1 - 1, p1 + 2):
            for k in range(p2 - 1, p2 + 2):
                count += grid[i, j, k]

    count -= grid[p0, p1, p2]
    return count

class Test17(unittest.TestCase):
    matrix = [".#.",
              "..#",
              "###"]


    def test(self):
        m = get_matrix(self.matrix, life_dict)
        self.assertEqual([[0,1,0],[0,0,1],[1,1,1]], m)

        grid, start = build_grid(m, 2)
        p0, p1, p2 = start
        self.assertEqual(m[0][0], grid[p0, p1, p2])

        self.assertEqual(1, check_neighbours((p0, p1, p2), grid))
        self.assertEqual(5, check_neighbours((p0, p1 + 1, p2 + 1), grid))
        self.assertEqual(3, check_neighbours((p0, p1 + 2, p2 + 1), grid))


def day17():
    m = get_matrix([l.strip() for l in read_lines("day17input.txt")], life_dict)
    grid, start = build_grid(m, 6)
    p0, p1, p2 = start
    dims = (1, 3, 3)

    start_time = time.time()

    new_grid = np.zeros((13, 15, 15), dtype=int)

#    for cycle in range(0, 6):
    for i in range(p0, p0 + dims[0]):
        for j in range(p1, p1 + dims[1]):
            for k in range(p2, p2 + dims[2]):
                count = check_neighbours((i, j, k), grid)
                if grid[i,j,k]:
                    new_grid[i,j,k] = not (count in range(2, 4))
                else:
                    new_grid[i,j,k] = count == 3


    task1 = None
    task2 = None
    return time.time() - start_time, task1, task2
