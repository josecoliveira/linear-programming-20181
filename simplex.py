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

    @staticmethod
    def __to_string_matrix_with_fractions(matrix):
        """
        Create a string with all values on the matrix. Each row is separated by a new line and each value is separated
        by a space.
        :param matrix: Numpy matrix.
        :type matrix: np.matrixlib.defmatrix.matrix
        :return: String representing the matrix.
        :rtype: str
        """
        string = ""
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                string += str(Fraction(matrix[i, j]).numerator) + "/" + str(
                    Fraction(matrix[i, j]).denominator) + " "
            string += "\n"
        return string

    @staticmethod
    def to_string_vector_with_fractions(matrix):
        """
        Create a string with all values on the vector. Each value is separated by comma and space.
        :param matrix: Numpy matrix with only one row.
        :type matrix: np.matrixlib.defmatrix.matrix
        :return: String representing the vector.
        :rtype: str
        """
        string = "["
        for i in range(matrix.shape[1]):
            string += str(matrix[0, i].numerator) + "/" + str(matrix[0, i].denominator)
            if i is not matrix.shape[1] - 1:
                string += ", "
        string += "]"
        return string

    @staticmethod
    def __zeros_matrix(rows, columns):
        """
        Create a Numpy matrix with zeros and fractions.
        :param rows: Number os rows.
        :type rows: int
        :param columns: Number of columns.
        :type columns: int
        :return: A Numpy matrix with zeros.
        :rtype: np.matrixlib.defmatrix.matrix
        """
        matrix = np.matrix(np.zeros((rows, columns))).astype('object')
        for row in range(rows):
            for column in range(columns):
                matrix[row, column] = Fraction(matrix[row, column])
        return matrix

    @staticmethod
    def __identity_matrix(size):
        """
        Create a identity Numpy matrix with fractions.
        :param size: Number of rows and columns of the matrix.
        :type size: int
        :return: A identity numpy matrix with fractions.
        :rtype: np.matrixlib.defmatrix.matrix
        """
        identity = np.identity(size)
        identity = identity.astype('object')
        for row in range(size):
            for column in range(size):
                identity[row, column] = Fraction(identity[row, column])
        return identity

    @staticmethod
    def __pivot_matrix(matrix, row, column):
        """
        Pivot a Numpy matrix using Gaussian elimination.
        :param matrix: Numpy matrix to be pivoted.
        :type matrix: np.matrixlib.defmatrix.matrix
        :param row: Row to be pivoted.
        :type row: int
        :param column: Column to be pivoted.
        :type column: int
        :return: None.
        """
        matrix[row] = matrix[row] / matrix[row, column]
        for i in range(matrix.shape[0]):
            if i != row:
                matrix[i] = matrix[i] - matrix[i, column] * matrix[row]

    def __init__(self, matrix: np.matrixlib.defmatrix.matrix = None, ct=None, a=None, b=None, fpi=False, basis=None):
        """
        Constructor for Linear Programing. It can be instantiate with a Numpy matrix with Fraction in the following
        form:

        | c^T 0 |
        |  A  b |

        corresponding to

        max c^T x
        s.t Ax <= b
            x >= 0.

        :param matrix: Numpy matrix represented like above-mentioned form.
        :type matrix: np.matrixlib.defmatrix.matrix
        :param ct: Numpy matrix representing c^T
        :type ct: np.matrixlib.defmatrix.matrix
        :param a: Numpy matrix representing A
        :type a: np.matrixlib.defmatrix.matrix
        :param b: Numpy matrix representing b
        :type b: np.matrixlib.defmatrix.matrix
        :param fpi: True whether LP is in standard form of equalities or false otherwise.
        :type fpi: bool
        :param basis: Array representing where the positions on A the basis are, if it is needed.
        :type basis: list
        """
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
        if fpi and basis is not None:
            self.num_variables_from_tableau = self.num_variables
            self.__make_tableau(gap=False)
            self.basis = basis
            for i in range(len(self.basis)):
                self.__pivot_tableau(i, self.basis[i])
        else:
            self.num_variables_from_tableau = self.num_variables + self.num_restrictions
            self.__make_tableau(gap=True)
            self.__init_basis()

        self.infeasible = False
        self.unlimited = False

    def __init_basis(self):
        """
        Create a array representing where the positions on A the base are.
        :return: Array with basis.
        :rtype: list
        """
        self.basis = []
        for i in range(self.num_restrictions):
            self.basis.append(self.num_variables_from_tableau - self.num_restrictions + i)

    def __init_basis_fpi(self):
        """
        Create a array representing where the positions on A the base are in case of performing primal simplex with
        auxiliary linear programming.
        :return: Array with basis.
        :rtype: list
        """
        self.basis = []
        for i in range(self.num_restrictions):
            self.basis.append(self.num_variables - self.num_restrictions + i)

    def __make_tableau(self, gap):
        """
        Make a tableau matrix with all submatrix
        :param gap: Boolean value to indicate if linear programming is in standard form of equalities.
        :type gap: bool
        :return: None
        """
        rows = self.num_restrictions + 1
        columns = self.num_restrictions + self.num_variables_from_tableau + 1
        tableau = self.__zeros_matrix(rows, columns)
        identity = self.__identity_matrix(self.num_restrictions)

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
        tableau[1:self.num_restrictions + 1, self.num_restrictions + self.num_variables_from_tableau] = self.b

        # -c^T
        tableau[0, self.num_restrictions:self.num_restrictions + self.num_variables] = -self.ct

        self.tableau = tableau

    @property
    def minus_ct_from_tableau(self):
        """
        :return: Submatrix from tableau with -c^T.
        :rtype: np.matrixlib.defmatrix.matrix
        """
        return self.tableau[0, self.num_restrictions:self.num_restrictions + self.num_variables_from_tableau]

    @property
    def objective_value(self):
        """
        :return: Objective value from tableau.
        :rtype: int
        """
        return self.tableau[0, -1]

    @property
    def operations_matrix(self):
        """
        :return: Submatrix from tableau with operations matrix.
        :rtype: np.matrixlib.defmatrix.matrix
        """
        return self.tableau[1:1 + self.num_restrictions, 0:self.num_restrictions]

    @property
    def a_from_tableau(self):
        """
        :return: Submatrix from tableau with A.
        :rtype: np.matrixlib.defmatrix.matrix
        """
        return self.tableau[1:1 + self.num_restrictions,
               self.num_restrictions:self.num_restrictions + self.num_variables_from_tableau]

    @property
    def b_from_tableau(self):
        """
        :return: Submatrix from tableau with b.
        :rtype: np.matrixlib.defmatrix.matrix
        """
        return np.matrix(self.tableau[1:, -1])

    @property
    def solution(self):
        """
        Make a solution from tableau and basis.
        :return: A Numpy matrix in array form with the solution.
        :rtype: np.matrixlib.defmatrix.matrix
        """
        solution = self.__zeros_matrix(1, self.num_variables_from_tableau)
        for i in range(self.num_restrictions):
            solution[0, self.basis[i]] = self.b_from_tableau[i, 0]
        return solution

    def __create_certificate_of_unlimited(self, problematic_column):
        """
        Generate a certificate when a linear programming is unlimited when perform primal simplex.
        :param problematic_column: Column that indicates the limitlessness on tableau.
        :type problematic_column: int
        :return: None
        """
        certificate = self.__zeros_matrix(1, self.num_variables_from_tableau)
        certificate[0, problematic_column] = 1
        for i in range(self.num_restrictions):
            certificate[0, self.basis[i]] = - self.a_from_tableau[i, problematic_column]
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
        :return: Objective value if found or None if it is infeasible or unlimited.
        :rtype: int
        """
        for column in range(len(self.basis)):
            self.__pivot_tableau(column, self.basis[column])

        while True:
            is_optimal = True

            # Find a entry of b that can be pivoted.
            for row in range(self.num_restrictions):
                if self.b_from_tableau[row, 0] < 0:
                    # Check if can be unlimited
                    if all(self.a_from_tableau[row, i] >= 0 for i in range(self.num_variables_from_tableau)):
                        self.infeasible = True
                        self.certificate = self.operations_matrix[row, 0:self.num_restrictions]
                        return None
                    is_optimal = False
                    break

            # Check if it is optimal
            if is_optimal:
                self.certificate = self.tableau[0, 0:self.num_restrictions]
                return self.objective_value

            # Find with columns will be pivoted.
            column = None
            current_ratio = None
            for j in range(self.num_variables_from_tableau):
                if self.a_from_tableau[row, j] < 0:
                    ratio = self.minus_ct_from_tableau[0, j] / (-self.a_from_tableau[row, j])
                    if current_ratio is None or current_ratio > ratio:
                        column = j
                        current_ratio = ratio
            self.basis[row] = column
            self.__pivot_tableau(row, column)

    @property
    def __primal_simplex(self):
        """
        Perform primal simplex.
        :return: Objective value if found or None if it is unlimited.
        :rtype: int
        """
        for column in range(len(self.basis)):
            self.__pivot_tableau(column, self.basis[column])

        while True:
            is_optimal = True

            # Find a input from c^t that can be increased.
            for column in range(self.num_variables_from_tableau):
                if self.minus_ct_from_tableau[0, column] < 0:
                    # Check if can be unlimited.
                    if all(self.a_from_tableau[i, column] <= 0 for i in range(self.num_restrictions)):
                        self.unlimited = True
                        self.__create_certificate_of_unlimited(column)
                        return None
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
            self.basis[row] = column
            self.__pivot_tableau(row, column)

    @property
    def __primal_simplex_by_auxiliary(self):
        """
        Build a instance of a linear programming to find a basis for original linear programing. Then perform primal
        simplex.
        :return: Objective value if found or none if it is unlimited or infeasible.
        :rtype: int
        """
        # Remove non-negativity of restrictions.
        for row in range(self.num_restrictions):
            if self.b_from_tableau[row, 0] < 0:
                self.tableau[1 + row] = -self.tableau[1 + row]

        # Create new c^T for auxiliary
        new_ct = self.__zeros_matrix(1, self.num_variables_from_tableau + self.num_restrictions)
        for column in range(self.num_restrictions):
            new_ct[0, self.num_variables_from_tableau + column] = Fraction(-1, 1)

        # Create new A for auxiliary
        new_a = self.__zeros_matrix(self.num_restrictions, self.num_variables_from_tableau + self.num_restrictions)

        new_a[0:self.num_restrictions, 0:self.num_variables_from_tableau] = self.a_from_tableau

        new_a[0:self.num_variables_from_tableau,
        self.num_variables_from_tableau:self.num_variables_from_tableau + self.num_restrictions] = self.__identity_matrix(
            self.num_restrictions)

        # Create new basis for auxiliary
        new_basis = []
        for column in range(self.num_restrictions):
            new_basis.append(self.num_variables_from_tableau + column)

        auxiliary_pl = LinearProgramming(ct=new_ct, a=new_a, b=self.b_from_tableau, fpi=True, basis=new_basis)
        result = auxiliary_pl.simplex

        if auxiliary_pl.unlimited:
            self.certificate = auxiliary_pl.certificate
            self.unlimited = True
            return None
        elif result < 0:
            self.certificate = auxiliary_pl.tableau[0, 0:self.num_restrictions]
            self.infeasible = True
            return None
        else:
            self.basis = auxiliary_pl.basis
            return self.__primal_simplex

    @property
    def simplex(self):
        """
        Find which simplex type will be performed and return its result.
        :return: Objective value if found or -1 if the linear programming is infeasible or unlimited.
        :rtype: int
        """
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

    if linear_programming.infeasible:
        print("0")
        print(linear_programming.to_string_vector_with_fractions(linear_programming.certificate))
    elif linear_programming.unlimited:
        print("1")
        print(linear_programming.to_string_vector_with_fractions(linear_programming.certificate))
    else:
        print("2")
        print(linear_programming.to_string_vector_with_fractions(linear_programming.solution))
        print(linear_programming.objective_value)
        print(linear_programming.to_string_vector_with_fractions(linear_programming.certificate))

    input_file.close()


if __name__ == "__main__":
    main()
