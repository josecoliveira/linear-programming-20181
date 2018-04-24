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
    def num_variables(self) -> int:
        return self.a.shape[1]

    @property
    def num_restrictions(self) -> int:
        return self.a.shape[0]

    @property
    def minus_c_from_tableau(self) -> np.matrixlib.defmatrix.matrix:
        return self.tableau[0, self.num_restrictions:self.num_restrictions + self.num_variables]

    @property
    def objective_value(self) -> int:
        return self.tableau[0: -1]

    @property
    def matrix_operations(self) -> np.matrixlib.defmatrix.matrix:
        return self.tableau[1:1 + self.num_restrictions, 0:self.num_restrictions]

    @property
    def a_from_tableau(self) -> np.matrixlib.defmatrix.matrix:
        return self.tableau[1:self.num_variables,
               self.num_restrictions:self.num_restrictions + self.num_variables]

    @property
    def b_from_tableau(self) -> np.matrixlib.defmatrix.matrix:
        return np.matrix(self.tableau[1:, -1])

    def __make_tableau(self):
        rows = self.num_restrictions + 1
        columns = self.num_restrictions + self.num_variables + self.num_restrictions + 1
        tableau: np.matrixlib.defmatrix.matrix = np.matrix(np.zeros((rows, columns))).astype('object')
        print(type(tableau))
        print(tableau)
        for i in range(rows):
            for j in range(columns):
                tableau[i, j] = Fraction(tableau[i, j])
        identity = np.identity(self.num_restrictions)
        identity = identity.astype('object')
        for i in range(self.num_restrictions):
            for j in range(self.num_restrictions):
                identity[i, j] = Fraction(identity[i, j])
        tableau[1:self.num_restrictions + 1, 0:self.num_restrictions] = identity
        tableau[1:self.num_restrictions + 1,
        self.num_restrictions:self.num_restrictions + self.num_variables] = self.a
        tableau[1:self.num_restrictions + 1,
        self.num_restrictions + self.num_variables:2 * self.num_restrictions + self.num_variables] = identity
        tableau[1:self.num_restrictions + 1,
        2 * self.num_restrictions + self.num_variables:3 * self.num_restrictions + self.num_variables] = self.b
        tableau[0, self.num_restrictions:self.num_restrictions + self.num_variables] = self.cT
        self.tableau = tableau

    @staticmethod
    def __pivot_matrix(matrix: np.matrixlib.defmatrix.matrix, row: int, column: int):
        matrix[row] = matrix[row] / matrix[row, column]
        for i in range(matrix.shape[0]):
            if i != row:
                matrix[i] = matrix[i] - matrix[i, row] * matrix[row]

    def simplex(self):
        print(self.__to_string_matrix_with_fractions(self.tableau))
        print(self.__to_string_matrix_with_fractions(self.b_from_tableau))


def main():
    input_file = open(sys.argv[1], 'r')
    lines = int(input_file.readline().replace("\n", ""))
    columns = int(input_file.readline().replace("\n", ""))
    string_matrix = input_file.readline().replace("\n", "")
    matrix = json.loads(string_matrix)

    matrix = convert_matrix_to_fractions(matrix)

    pl = Pl(matrix)
    pl.simplex()

    input_file.close()


if __name__ == "__main__":
    main()
