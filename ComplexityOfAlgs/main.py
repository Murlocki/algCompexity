# task1
def multiple_vectors(vector1, vector2):
    s = 0
    for i in range(len(vector1)):
        a = vector1[i] * vector2[i]
        s = s + a
    return s


# 1 + 2n
# print(multiple_vectors([1, 2, 3], [1, 2, 3]))


def multiply_matrix(matrix1, matrix2):
    total_matrix = []
    lenght = len(matrix1)
    for i in range(lenght):
        row = []
        for j in range(lenght):
            b = [matrix2[o][j] for o in range(lenght)]
            a = multiple_vectors(matrix1[i], b)
            row.append(a)
        total_matrix.append(row)
    return total_matrix


# 2 + n * (2 + n * (3n+4))
# 2 + 2n + 4n^2+ 3n^3
# print(multiply_matrix([[2,-3],
#                      [5,4]],
#                    [[-7,5],
#                    [2,-1],]))


def bubble(array):
    length = len(array)
    for i in range(length - 1):
        for j in range(i + 1, length):
            if array[i] < array[j]:
                array[i], array[j] = array[j], array[i]

    return array


# 1 + (n-1) * (n-2+1)*(n-1)/2 * (1+1)
# 1 + (n-1) * (n-1)*(n-1)
# print(bubble([4,5,3,2,1]))


def choice_sort(array):
    length = len(array)
    for i in range(length - 1):
        min_index = i
        for k in range(i + 1, length):
            if array[k] < array[min_index]:
                min_index = k
        array[i], array[min_index] = array[min_index], array[i]
    return array


print(choice_sort([4, 5, 3, 2, 66, 5, 1]))


# 1 + (n-1) * (1 + (n-2+1)/2 * (n-1) * (1+1) + 1)
# 1 + (n-1) * (2 + (n-1)^2)

def input_sort(array):
    length = len(array)
    for i in range(2, length):
        key = array[i]
        j = i - 1
        while j >= 0 and array[j] > key:
            array[j + 1], array[j] = array[j], array[j + 1]
            j = j - 1
        array[j + 1] = key
    return array


print(input_sort([4, 5, 3, 2, 66, 5, 1]))
# 1 + (n-1) * (1+ 1 +