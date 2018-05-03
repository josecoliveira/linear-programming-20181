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


class LinearProgramming:

    def __init__(self, matrix: np.matrixlib.defmatrix.matrix = None, ct=None, a=None, b=None, fpi=False):
        if matrix is not None:
            self.ct = matrix[0, 0:-1]
            self.a = matrix[1:, 0:-1]
            self.b = matrix[1:, -1]
        elif ct is not None and a is not None and b is not None:
            self.ct = ct
            self.a = a
            self.b = b
        self.num_variables = self.a.shape[1]
        self.num_restrictions = self.a.shape[0]
        if fpi:
            self.__make_tableau(gap=False)
            self.__init_base()
            for i in range(len(self.base)):
                self.__pivot_tableau(i, self.base[i])
            self.num_variables_from_tableau = self.num_variables
        else:
            self.num_variables_from_tableau = self.num_variables + self.num_restrictions
            self.__make_tableau(gap=True)
            self.__init_base()

        self.impossible = False
        self.unlimited = False

    def __init_base(self):
        self.base = []
        for i in range(self.num_restrictions):
            self.base.append(self.num_variables_from_tableau - self.num_restrictions + i)

    def __init_base_fpi(self):
        self.base = []
        for i in range(self.num_restrictions):
            self.base.append(self.num_variables - self.num_restrictions + i)

    @staticmethod
    def __to_string_matrix_with_fractions(matrix: np.matrixlib.defmatrix.matrix) -> str:
        string = ""
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                string += str(Fraction(matrix[i, j]).numerator) + "/" + str(
                    Fraction(matrix[i, j]).denominator) + " "
            string += "\n"
        return string

    @staticmethod
    def to_string_vector_with_fractions(matrix: np.matrixlib.defmatrix.matrix) -> str:
        string = "["
        for i in range(matrix.shape[1]):
            string += str(matrix[0, i].numerator) + "/" + str(matrix[0, i].denominator)
            if i is not matrix.shape[1] - 1:
                string += " "
        string += "]"
        return string

    @property
    def minus_ct_from_tableau(self) -> np.matrixlib.defmatrix.matrix:
        return self.tableau[0, self.num_restrictions:self.num_restrictions + self.num_variables_from_tableau]

    @property
    def objective_value(self) -> int:
        return self.tableau[0, -1]

    @property
    def operations_matrix(self) -> np.matrixlib.defmatrix.matrix:
        return self.tableau[1:1 + self.num_restrictions, 0:self.num_restrictions]

    @property
    def a_from_tableau(self) -> np.matrixlib.defmatrix.matrix:
        return self.tableau[1:self.num_variables,
               self.num_restrictions:self.num_restrictions + self.num_variables_from_tableau]

    @property
    def b_from_tableau(self) -> np.matrixlib.defmatrix.matrix:
        return np.matrix(self.tableau[1:, -1])

    @property
    def solution(self):
        """
        Make a solution from tableau and basis.
        :return: A Numpy matrix in array form with the solution.
        :rtype: np.matrixlib.defmatrix.matrix
        """
        solution = np.matrix(np.zeros((1, self.num_variables_from_tableau))).astype('object')
        for i in range(self.num_variables_from_tableau):
            solution[0, i] = Fraction(solution[0, i])
        for i in range(self.num_restrictions):
            solution[0, self.base[i]] = self.b_from_tableau[i, 0]
        return solution

    def __make_tableau(self, gap):
        """
        Make a tableau matrix with all submatrix
        :param gap: Boolean value to indicate if linear programming is in standard form of equalities.
        :type gap: bool
        :return: None
        """
        rows = self.num_restrictions + 1
        columns = self.num_restrictions + self.num_variables_from_tableau + 1
        tableau = np.matrix(np.zeros((rows, columns))).astype('object')
        for i in range(rows):
            for j in range(columns):
                tableau[i, j] = Fraction(tableau[i, j])
        identity = np.identity(self.num_restrictions)
        identity = identity.astype('object')
        for i in range(self.num_restrictions):
            for j in range(self.num_restrictions):
                identity[i, j] = Fraction(identity[i, j])

        # Operations matrix
        tableau[1:self.num_restrictions + 1, 0:self.num_restrictions] = identity

        # A
        tableau[1:self.num_restrictions + 1,
        self.num_restrictions:self.num_restrictions + self.num_variables] = self.a

        # Gap matrix
        if gap:
            tableau[1:self.num_restrictions + 1,
            self.num_restrictions + self.num_variables:2 * self.num_restrictions + self.num_variables] = identity

        # b
        tableau[1:self.num_restrictions + 1,
        2 * self.num_restrictions + self.num_variables:3 * self.num_restrictions + self.num_variables] = self.b

        # -c^T
        tableau[0, self.num_restrictions:self.num_restrictions + self.num_variables] = -self.ct

        self.tableau = tableau

    @staticmethod
    def __pivot_matrix(matrix, row, column):
        """
        Pivot a Numpy matrix using Gaussian elimination.
        :param matrix: Numpy matrix to be pivoted.
        :type matrix: np.matrixlib.defmatrix.matrix
        :param row: Row to be pivoted
        :type row: int
        :param column: Column to be pivoted
        :type column: int
        :return: None
        """
        matrix[row] = matrix[row] / matrix[row, column]
        for i in range(matrix.shape[0]):
            if i != row:
                matrix[i] = matrix[i] - matrix[i, column] * matrix[row]

    def __create_certificate_of_unlimited(self, problematic_column):
        """
        Generate a certificate when a linear programming is unlimited.
        :param problematic_column: Column that indicates the limitlessness on tableau.
        :type problematic_column: int
        :return: None
        """
        certificate = np.matrix(np.zeros((1, self.num_variables_from_tableau))).astype('object')
        for i in range(self.num_variables_from_tableau):
            certificate[0, i] = Fraction(certificate[0, i])
        certificate[0, problematic_column] = 1
        for i in range(self.num_restrictions):
            certificate[0, self.base[i]] = - self.a_from_tableau[i, problematic_column]
        self.certificate = certificate

    def __pivot_tableau(self, a_row, a_column):
        """
        Call __pivot_matrix to pivot an entry of A matrix on tableau.
        :param a_row: Row to be pivoted on A matrix.
        :param a_column: Column to be pivoted on A matrix
        :return: None
        """
        self.__pivot_matrix(self.tableau, 1 + a_row, self.num_restrictions + a_column)

    @property
    def __dual_simplex(self):
        """
        Perform dual simplex.
        :return: Objective value if found or -1 if the linear programming is infeasible or unlimited.
        :rtype: int
        """
        while True:
            for i in range(self.num_restrictions):
                if self.b_from_tableau[i, 0] < 0:
                    break
                else:
                    return self.objective_value
            column = None
            current_ratio = None
            for j in range(self.num_variables_from_tableau):
                ratio = ((-1) * self.minus_ct_from_tableau[0, j]) / self.a_from_tableau[i, j]
                if self.a_from_tableau[i, j] < 0 and (current_ratio < ratio or current_ratio is None):
                    column = j
                    current_ratio = ratio
            self.__pivot_matrix(self.tableau, 1 + i, self.num_restrictions + column)
            self.base[i] = column

    @property
    def __primal_simplex(self):
        """
        Perform primal simplex.
        :return: Objective value if found or -1 if the linear programming is unlimited.
        :rtype: int
        """
        while True:
            is_optimal = True

            # Find a input from c^t that can be increased.
            for column in range(self.num_variables_from_tableau):
                if self.minus_ct_from_tableau[0, column] < 0:
                    # Check if can be unlimited.
                    if all(self.a_from_tableau[i, column] <= 0 for i in range(self.num_restrictions)):
                        self.unlimited = True
                        self.__create_certificate_of_unlimited(column)
                        return -1
                    is_optimal = False
                    break

            # Check if it is optimal.
            if is_optimal:
                self.certificate = self.tableau[0, 0:self.num_restrictions]
                return self.objective_value

            # Find with row will be pivoted.
            row = None
            current_ratio = None
            for j in range(self.num_restrictions):
                if self.a_from_tableau[j, column] > 0:
                    ratio = self.b_from_tableau[j, 0] / self.a_from_tableau[j, column]
                    if current_ratio is None or current_ratio > ratio:
                        row = j
                        current_ratio = ratio
            self.base[row] = column
            self.__pivot_tableau(row, column)

            print(self.__to_string_matrix_with_fractions(self.tableau))

    @property
    def __primal_simplex_by_auxiliary(self):
        print("By Auxiliary")
        for row in range(self.num_restrictions):
            if self.b_from_tableau[row, 0] < 0:
                self.tableau[1 + row] = -self.tableau[1 + row]
        print(self.__to_string_matrix_with_fractions(self.tableau))
        print(self.__to_string_matrix_with_fractions(self.a_from_tableau))
        print(self.__to_string_matrix_with_fractions(self.b_from_tableau))
        return -2

    @property
    def simplex(self):
        """
        Find which simplex type will be performed e return its result.
        :return: Objective value if found or -1 if the linear programming is infeasible or unlimited.
        :rtype: int
        """
        print("SIMPLEX")
        if any(self.b_from_tableau[i, 0] < 0 for i in range(self.num_restrictions)) and all(
                self.minus_ct_from_tableau[0, i] >= 0 for i in range(self.num_variables_from_tableau)):
            return self.__dual_simplex
        elif all(self.b_from_tableau[i, 0] >= 0 for i in range(self.num_restrictions)) and any(
                self.minus_ct_from_tableau[0, i] < 0 for i in range(self.num_variables_from_tableau)):
            return self.__primal_simplex
        elif any(self.b_from_tableau[i, 0] < 0 for i in range(self.num_restrictions)) and any(
                self.minus_ct_from_tableau[0, i] < 0 for i in range(self.num_variables_from_tableau)):
            return self.__primal_simplex_by_auxiliary


def main():
    input_file = open(sys.argv[1], 'r')
    lines = int(input_file.readline().replace("\n", ""))
    columns = int(input_file.readline().replace("\n", ""))
    string_matrix = input_file.readline().replace("\n", "")
    matrix = json.loads(string_matrix)

    matrix = convert_matrix_to_fractions(matrix)

    linear_programming: LinearProgramming = LinearProgramming(matrix=matrix)

    result = linear_programming.simplex

    if result is -1:
        if linear_programming.unlimited:
            print("1")
            print(linear_programming.to_string_vector_with_fractions(linear_programming.certificate))
        elif linear_programming.impossible:
            print("Inviável")
    elif result is -2:
        print("Debugging")
        pass
    else:
        print("2")
        print(linear_programming.to_string_vector_with_fractions(linear_programming.solution))
        print(linear_programming.objective_value)
        print(linear_programming.to_string_vector_with_fractions(linear_programming.certificate))

    input_file.close()


if __name__ == "__main__":
    main()
