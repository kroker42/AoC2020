# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# read one int per line from file
def read_ints(fname):
    return [int(line) for line in open(fname)]
    #with open(fname) as f:
        #        array = []
    #   for line in f:
    #       array.append(int(line))
#   return array

def find_2020_sum(array):
    for i in range(len(array)-1):
        for j in array[i+1:]:
            if array[i] + j == 2020:
                return array[i] * j

def test_find_2020_sum():
    test_set = [1721, 979, 366, 299, 675, 1456]
    result = 1721 * 299
    print("test_find_2020_sum: " + str(result == find_2020_sum(test_set)))

def find_2020_sum3(array):
    for i in range(len(array)-1):
        for j in range(i+1, len(array) - 1):
            for k in array[j+1:] :
                if array[i] + array[j] + k == 2020:
                    return array[i] * array[j] * k

def test_find_2020_sum3():
    test_set = [1721, 979, 366, 299, 675, 1456]
    result = 979 * 366 * 675
    print("test_find_2020_sum3: " + str(result == find_2020_sum3(test_set)))

def day1():
    expenses = read_ints('day1input.txt')
    print(find_2020_sum(expenses))
    print(find_2020_sum3(expenses))

def test():
    test_find_2020_sum()
    test_find_2020_sum3()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()
    day1()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
