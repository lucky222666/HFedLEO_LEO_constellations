fraction0 = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
clients = [
    [22, 27, 24, 23, 43, 28, 9, 42, 38, 8, 6, 63, 7, 64, 36, 39, 5, 46, 65, 11, 12, 16, 59, 62, 10, 25, 45, 19, 61, 21, 26, 31, 18, 35, 44, 41, 20, 40, 30, 15, 37, 33, 29, 13, 34, 32, 60, 47, 17, 66, 14, 48, 67, 49, 68, 50, 69, 51, 52, 70, 71, 53, 0, 54, 1, 55, 2, 56, 3, 4, 58, 57],
    [55, 52, 2, 39, 54, 20, 1, 53, 69, 4, 37, 22, 38, 21, 36, 23, 35, 41, 5, 42, 46, 0, 68, 57, 60, 40, 50, 3, 51, 56, 70, 65, 61, 19, 63, 48, 59, 45, 24, 49, 67, 43, 64, 58, 44, 18, 47, 6, 66, 62, 17, 7, 25, 8, 27, 10, 28, 11, 29, 12, 30, 13, 31, 14, 32],
    [4, 62, 57, 67, 66, 61, 30, 60, 31, 12, 46, 11, 29, 44, 47, 28, 48, 43, 49, 5, 42, 52, 53, 45, 3, 50, 63, 27, 71, 14, 58, 10, 9, 8, 59, 2, 6, 56, 65, 32, 7, 51, 70, 64, 54, 55, 26, 1, 69, 0, 68, 15, 33, 13, 16, 34, 17, 35],
    [7, 65, 60, 70, 66, 69, 33, 34, 15, 49, 14, 32, 50, 31, 51, 64, 46, 4, 8, 55, 45, 68, 63, 67, 10, 6, 48, 53, 61, 17, 13, 11, 30, 2, 12, 62, 9, 5, 0, 47, 59, 35, 16, 54, 36, 52, 71, 1, 57, 58, 29],
    [70, 65, 61, 8, 35, 69, 33, 48, 32, 50, 15, 47, 51, 31, 5, 52, 46, 4, 71, 56, 34, 49, 68, 16, 3, 64, 63, 11, 7, 60, 67, 10, 6, 13, 66, 9, 12, 14, 62, 1, 18, 55, 17, 53],
    [64, 69, 12, 8, 65, 14, 66, 46, 9, 51, 70, 61, 4, 33, 49, 48, 34, 50, 6, 67, 47, 16, 35, 53, 30, 54, 59, 58, 32, 13, 63, 62, 29, 15, 10, 11],
    [65, 13, 61, 70, 8, 71, 30, 35, 50, 69, 33, 48, 34, 32, 15, 51, 67, 47, 52, 5, 46, 0, 4, 57, 49, 68, 64, 11, 7],
    [70, 65, 7, 13, 66, 64, 16, 14, 62, 15, 10, 47, 33, 34, 5, 49, 50, 32, 51, 35, 31, 69],
    [64, 67, 66, 13, 9, 65, 51, 70, 12, 50, 49, 33, 34, 48, 7],
    [66, 61, 70, 13, 8, 65, 71, 67],
]

orbits = [i for i in range(72)]
trained_orbits = set([])

def coverage(round):
    t = 0
    while t < round:
        for o in clients[t]:
            trained_orbits.add(o)
        t = t + 1
    print(len(trained_orbits) / len(orbits))


if __name__ == '__main__':
    coverage(5)

