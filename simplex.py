import json
import sys
from fractions import Fraction

import numpy as np
from numpy.core.multiarray import ndarray


class Pl:

    def __init__(self, matrix):
        self.cT = matrix[0, 0:-1]
        self.a = matrix[1:, 0:-1]
        self.b = matrix[1:, -1]

    @staticmethod
    def __to_string_matrix_with_fractions(matrix):
        string = ""
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                string += str(Fraction(matrix[i, j]).numerator) + "/" + str(
                    Fraction(matrix[i, j]).denominator) + " "
            string += "\n"
        return string

    def get_num_variables(self):
        return self.a.shape[1]

    def get_num_restrictions(self) -> int:
        return self.a.shape[0]

    def make_tableau(self):
        rows = self.get_num_restrictions() + 1
        columns = self.get_num_restrictions() + self.get_num_variables() + self.get_num_restrictions() + 1
        tableau = np.zeros((rows, columns))
        tableau: ndarray = tableau.astype('object')
        for i in range(rows):
            for j in range(columns):
                tableau[i, j] = Fraction(tableau[i, j])
        identity = np.identity(self.get_num_restrictions())
        identity = identity.astype('object')
        for i in range(self.get_num_restrictions()):
            for j in range(self.get_num_restrictions()):
                identity[i, j] = Fraction(identity[i, j])
        tableau[1:self.get_num_restrictions() + 1, 0:self.get_num_restrictions()] = identity
        tableau[1:self.get_num_restrictions() + 1,
        self.get_num_restrictions():self.get_num_restrictions() + self.get_num_variables()] = self.a
        tableau[1:self.get_num_restrictions() + 1,
        self.get_num_restrictions() + self.get_num_variables():2 * self.get_num_restrictions() + self.get_num_variables()] = identity
        tableau[1:self.get_num_restrictions() + 1,
        2 * self.get_num_restrictions() + self.get_num_variables():3 * self.get_num_restrictions() + self.get_num_variables()] = self.b
        tableau[0, self.get_num_restrictions():self.get_num_restrictions() + self.get_num_variables()] = self.cT
        return tableau

    def simplex(self):
        tableaux = self.make_tableau()
        print(self.__to_string_matrix_with_fractions(tableaux))


def convert_matrix(matrix):
    matrix = np.matrix(matrix)
    matrix = matrix.astype('object')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = Fraction(matrix[i, j])
    return matrix


def main():
    inputFile = open(sys.argv[1], 'r')
    lines = int(inputFile.readline().replace("\n", ""))
    columns = int(inputFile.readline().replace("\n", ""))
    stringMatrix = inputFile.readline().replace("\n", "")
    matrix = json.loads(stringMatrix)

    matrix = convert_matrix(matrix)

    print(matrix)

    pl = Pl(matrix)
    # print(pl.cT)

    # print(pl.getNumVariables())

    for x in range(pl.get_num_restrictions()):
        for y in range(pl.get_num_variables()):
            print(str(x) + ", " + str(y) + ": " + str(pl.a[x, y]))

    pl = Pl(matrix)
    pl.simplex()

    inputFile.close()


if __name__ == "__main__":
    main()
