import json
import sys
from fractions import Fraction

import numpy as np


def convert_matrix_to_fractions(matrix: np.matrixlib.defmatrix.matrix) -> np.matrixlib.defmatrix.matrix:
    matrix = np.matrix(matrix)
    matrix: np.matrixlib.defmatrix.matrix = matrix.astype('object')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = Fraction(matrix[i, j])
    return matrix


class Pl:

    def __init__(self, matrix: np.matrixlib.defmatrix.matrix):
        self.cT = matrix[0, 0:-1]
        self.a = matrix[1:, 0:-1]
        self.b = matrix[1:, -1]
        self.__make_tableau()

    @staticmethod
    def __to_string_matrix_with_fractions(matrix: np.matrixlib.defmatrix.matrix) -> str:
        string = ""
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                string += str(Fraction(matrix[i, j]).numerator) + "/" + str(
                    Fraction(matrix[i, j]).denominator) + " "
            string += "\n"
        return string

    @property
    def get_num_variables(self) -> int:
        return self.a.shape[1]

    @property
    def get_num_restrictions(self) -> int:
        return self.a.shape[0]

    def __make_tableau(self):
        rows = self.get_num_restrictions + 1
        columns = self.get_num_restrictions + self.get_num_variables + self.get_num_restrictions + 1
        tableau = np.zeros((rows, columns))
        tableau: np.matrixlib.defmatrix.matrix = tableau.astype('object')
        for i in range(rows):
            for j in range(columns):
                tableau[i, j] = Fraction(tableau[i, j])
        identity = np.identity(self.get_num_restrictions)
        identity = identity.astype('object')
        for i in range(self.get_num_restrictions):
            for j in range(self.get_num_restrictions):
                identity[i, j] = Fraction(identity[i, j])
        tableau[1:self.get_num_restrictions + 1, 0:self.get_num_restrictions] = identity
        tableau[1:self.get_num_restrictions + 1,
        self.get_num_restrictions:self.get_num_restrictions + self.get_num_variables] = self.a
        tableau[1:self.get_num_restrictions + 1,
        self.get_num_restrictions + self.get_num_variables:2 * self.get_num_restrictions + self.get_num_variables] = identity
        tableau[1:self.get_num_restrictions + 1,
        2 * self.get_num_restrictions + self.get_num_variables:3 * self.get_num_restrictions + self.get_num_variables] = self.b
        tableau[0, self.get_num_restrictions:self.get_num_restrictions + self.get_num_variables] = self.cT
        self.tableau = tableau

    @staticmethod
    def __pivot_matrix(matrix: np.matrixlib.defmatrix.matrix, row: int, column: int):
        matrix[row] = matrix[row] / matrix[row, column]
        for i in range(matrix.shape[0]):
            if i != row:
                matrix[i] = matrix[i] - matrix[i, row] * matrix[row]

    def simplex(self):
        print(self.__to_string_matrix_with_fractions(self.tableau))


def main():
    input_file = open(sys.argv[1], 'r')
    lines = int(input_file.readline().replace("\n", ""))
    columns = int(input_file.readline().replace("\n", ""))
    stringMatrix = input_file.readline().replace("\n", "")
    matrix = json.loads(stringMatrix)

    matrix = convert_matrix_to_fractions(matrix)
    print(matrix)

    pl = Pl(matrix)
    # print(pl.cT)

    # print(pl.getNumVariables())

    # for x in range(pl.get_num_restrictions):
    #     for y in range(pl.get_num_variables):
    #         print(str(x) + ", " + str(y) + ": " + str(pl.a[x, y]))
    #
    # pl = Pl(matrix)
    # pl.simplex()

    input_file.close()

    # x = np.matrix


if __name__ == "__main__":
    main()
