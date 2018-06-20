import json
import sys
from fractions import Fraction
from math import floor
from math import ceil

import numpy as np

"""

Trabalho Prático de Pesquisa Operacional

José Carlos de Oliveira Júnior

18 de junho de 2018

Este trabalho deve ser executado em uma versão 3.x do Python.
O comando para executá-lo é

& python simplex.py input.txt

Sendo "input.txt" o arquivo de entrada.

"""


def convert_matrix_to_fractions(matrix):
    """
    :param matrix: Numpy matrix
    :type matrix: np.matrixlib.defmatrix.matrix
    :return: Numpy matrix with Fractions.
    :rtype: np.matrixlib.defmatrix.matrix
    """
    matrix = np.matrix(matrix)
    matrix = matrix.astype('object')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = Fraction(matrix[i, j])
    return matrix


class LinearProgramming:
    tableau: np.matrixlib.defmatrix.matrix = None
    LESS = 0
    GREATER = 1

    @staticmethod
    def __to_string_fraction(fraction):
        """

        :param fraction:
        :type fraction: fractions.Fraction
        :return:
        :rtype: fraction.Fractions
        """
        return str(fraction.numerator) + "/" + str(fraction.denominator)

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
        sizes = [None] * matrix.shape[1]

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                size = len(LinearProgramming.__to_string_fraction(matrix[i, j]))
                if sizes[j] is None or sizes[j] < size:
                    sizes[j] = size

        string = ""
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                new_string = str(Fraction(matrix[i, j]).numerator) + "/" + str(Fraction(matrix[i, j]).denominator)
                string += (" " * (sizes[j] - len(new_string))) + new_string + " "
            if i is not matrix.shape[0] - 1:
                string += "\n"
        return string

    def __to_string_tableau(self):
        sizes = [None] * self.tableau.shape[1]

        for i in range(self.tableau.shape[0]):
            for j in range(self.tableau.shape[1]):
                size = len(self.__to_string_fraction(self.tableau[i, j]))
                if sizes[j] is None or sizes[j] < size:
                    sizes[j] = size

        string = ""
        for j in range(self.num_restrictions):
            new_string = self.__to_string_fraction(self.tableau[0, j])
            string += (" " * (sizes[j] - len(new_string))) + new_string + " "
        string += "│ "
        for j in range(self.num_restrictions, self.num_restrictions + self.num_variables_from_tableau):
            new_string = self.__to_string_fraction(self.tableau[0, j])
            string += (" " * (sizes[j] - len(new_string))) + new_string + " "
        string += "│ "
        j += 1
        new_string = self.__to_string_fraction(self.tableau[0, j])
        string += (" " * (sizes[j] - len(new_string))) + new_string + " \n"

        for j in range(self.num_restrictions):
            string += ("─" * (sizes[j] + 1))
        string += "┼"
        for j in range(self.num_restrictions, self.num_restrictions + self.num_variables_from_tableau):
            string += ("─" * (sizes[j] + 1))
        string += "─┼" + ("─" * (sizes[j + 1] + 1)) + "\n"

        for i in range(1, self.tableau.shape[0]):
            for j in range(self.num_restrictions):
                new_string = self.__to_string_fraction(self.tableau[i, j])
                string += (" " * (sizes[j] - len(new_string))) + new_string + " "
            string += "│ "
            for j in range(self.num_restrictions, self.num_restrictions + self.num_variables_from_tableau):
                new_string = self.__to_string_fraction(self.tableau[i, j])
                string += (" " * (sizes[j] - len(new_string))) + new_string + " "
            string += "│ "
            j += 1
            new_string = self.__to_string_fraction(self.tableau[i, j])
            string += (" " * (sizes[j] - len(new_string))) + new_string
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

    def __init__(self, matrix=None, ct=None, a=None, b=None, fpi=False, basis=None, log_file=None,
                 operations_matrix=None):
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
        :param log_file: File to print tableau for each step.
        :type log_file: _io.TextIO
        """
        self.log_file = log_file

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

            if operations_matrix is not None:
                self.tableau[1:1 + self.num_restrictions, 0:self.num_restrictions] = operations_matrix

            self.basis = basis
            for i in range(len(self.basis)):
                self.__pivot_tableau(i, self.basis[i])
        else:
            self.num_variables_from_tableau = self.num_variables + self.num_restrictions
            self.__make_tableau(gap=True)
            self.__init_basis()
            # self.__print_on_log("Basis: " + str(self.basis) + "\n\n")

        self.infeasible = False
        self.unlimited = False

        # Branch and bound
        self.integer_solution = None
        self.integer_objective_value = Fraction(0)
        self.integer_found = False

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

    def __print_on_log(self, string):
        """
        Print a string on log file if is defined.
        :param string: String to be printed.
        :type string: str
        """
        if self.log_file is not None:
            self.log_file.write(string)

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

    def __dual_simplex(self):
        """
        Perform dual simplex.
        :return: Objective value if found or None if it is infeasible or unlimited.
        :rtype: int
        """
        # self.__print_on_log("Dual simplex\n\n")

        step = 1

        # self.__print_on_log("Tableau #" + str(step) + "\n")
        # self.__print_on_log(self.__to_string_tableau() + '\n\n')

        for column in range(len(self.basis)):
            self.__pivot_tableau(column, self.basis[column])

        while True:
            step += 1
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

            # self.__print_on_log("Tableau #" + str(step) + "\n")
            # self.__print_on_log(self.__to_string_tableau() + "\n\n")

    def __primal_simplex(self):
        """
        Perform primal simplex.
        :return: Objective value if found or None if it is unlimited.
        :rtype: int
        """
        step = 1

        # self.__print_on_log("Primal simplex\n\n")

        for column in range(len(self.basis)):
            self.__pivot_tableau(column, self.basis[column])

        # self.__print_on_log("Tableau #" + str(step) + "\n")
        # self.__print_on_log(self.__to_string_tableau() + "\n\n")

        while True:
            step += 1
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

            # self.__print_on_log("Tableau #" + str(step) + "\n")
            # self.__print_on_log(self.__to_string_tableau() + "\n\n")

    def __primal_simplex_by_auxiliary(self):
        """
        Build a instance of a linear programming to find a basis for original linear programing. Then perform primal
        simplex.
        :return: Objective value if found or None if it is unlimited or infeasible.
        :rtype: int
        """
        # self.__print_on_log("Auxiliary Linear Programming\n\n")

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

        auxiliary_pl = LinearProgramming(ct=new_ct, a=new_a, b=self.b_from_tableau, fpi=True, basis=new_basis,
                                         log_file=self.log_file, operations_matrix=self.operations_matrix)
        # self.__print_on_log(self.__to_string_tableau() + "\n\n")
        result = auxiliary_pl.simplex()

        if auxiliary_pl.unlimited:
            self.certificate = auxiliary_pl.certificate
            self.unlimited = True
            return None
        elif result < 0:
            self.certificate = auxiliary_pl.tableau[0, 0:self.num_restrictions]
            self.infeasible = True
            return None
        else:
            self.basis = list(auxiliary_pl.basis)
            return self.__primal_simplex()

    def simplex(self):
        """
        Find which simplex type will be performed and return its result.
        :return: Objective value if found or None if the linear programming is infeasible or unlimited.
        :rtype: int
        """
        if any(self.b_from_tableau[i, 0] < 0 for i in range(self.num_restrictions)) and all(
                self.minus_ct_from_tableau[0, i] >= 0 for i in range(self.num_variables_from_tableau)):
            return self.__dual_simplex()
        elif all(self.b_from_tableau[i, 0] >= 0 for i in range(self.num_restrictions)) and any(
                self.minus_ct_from_tableau[0, i] < 0 for i in range(self.num_variables_from_tableau)):
            return self.__primal_simplex()
        elif any(self.b_from_tableau[i, 0] < 0 for i in range(self.num_restrictions)) and any(
                self.minus_ct_from_tableau[0, i] < 0 for i in range(self.num_variables_from_tableau)):
            return self.__primal_simplex_by_auxiliary()

    def addRestriction(self, restriction):
        """
        Add new restriction to tableau
        :param restriction:
        :type restriction: np.matrixlib.defmatrix.matrix
        """

        # print(self.__to_string_tableau())

        new_tableau: np.matrixlib.defmatrix.matrix = self.__zeros_matrix(self.tableau.shape[0] + 1,
                                                                         self.tableau.shape[1] + 2)

        # Certificate vector
        new_tableau[0, 0:self.num_restrictions] = self.tableau[0, 0:self.num_restrictions]

        # Minus c_T from tableau
        new_tableau[0,
        self.num_restrictions + 1:self.num_restrictions + 1 + self.num_variables_from_tableau] = self.minus_ct_from_tableau

        # Objective value
        new_tableau[0, -1] = self.objective_value

        # Operations matrix
        new_tableau[1:1 + self.num_restrictions, 0:self.num_restrictions] = self.operations_matrix
        new_tableau[1 + self.num_restrictions, self.num_restrictions] = Fraction(1)

        # A from tableau
        new_tableau[1:1 + self.num_restrictions,
        self.num_restrictions + 1:self.num_restrictions + 1 + self.num_variables_from_tableau] = self.a_from_tableau
        new_tableau[-1,
        self.num_restrictions + 1:self.num_restrictions + 1 + self.num_variables_from_tableau] = restriction[0,
                                                                                                 0:self.num_variables_from_tableau]
        new_tableau[-1, -2] = Fraction(1)

        # b from tableau
        new_tableau[1:self.num_restrictions + 1, -1] = self.b_from_tableau
        new_tableau[-1, -1] = restriction[0, -1]

        self.num_variables_from_tableau += 1
        self.num_restrictions += 1
        self.tableau = new_tableau
        old_basis = np.copy(self.basis)
        self.basis.append(self.a_from_tableau.shape[1] - 1)

        # self.__print_on_log("New tableau\n\n")
        # self.__print_on_log(self.__to_string_tableau() + "\n\n")

        # self.__print_on_log("Antes de repivotar o novo tableau para conferir restrição\n")
        # self.__print_on_log(self.__to_string_tableau())
        # self.__print_on_log(str(self.basis) + "\n")

        for i in range(self.num_restrictions - 1):
            self.__pivot_tableau(i, self.basis[i])

        # self.__print_on_log("Antes do simplex de novo tableau\n")
        # self.__print_on_log(self.__to_string_tableau())
        # self.__print_on_log(str(self.basis) + "\n")

        self.simplex()

    def remove_last_restriction(self):
        new_tableau: np.matrixlib.defmatrix.matrix = self.__zeros_matrix(self.tableau.shape[0] - 1,
                                                                         self.tableau.shape[1] - 2)

        # Certificate vector
        new_tableau[0, 0:self.num_restrictions - 1] = self.tableau[0, 0:self.num_restrictions - 1]

        # Minus c_T from tableau
        new_tableau[0, self.num_restrictions - 1:self.num_restrictions + self.num_variables_from_tableau - 2] = self.minus_ct_from_tableau[0, 0:self.num_variables_from_tableau - 1]

        # Objective value
        new_tableau[0, -1] = self.tableau[0, -1]

        # Operations Matrix
        new_tableau[1:, 0:self.num_restrictions - 1] = self.operations_matrix[0:-1, 0:-1]

        # A from tableau
        new_tableau[1:, self.num_restrictions - 1:self.num_restrictions + self.num_variables_from_tableau - 2] = self.a_from_tableau[0:self.num_restrictions - 1, 0:self.num_variables_from_tableau - 1]

        # b from tableau
        new_tableau[1:, -1] = self.b_from_tableau[0:self.num_restrictions - 1, -1]

        del self.basis[-1]

        self.num_restrictions -= 1
        self.num_variables_from_tableau -= 1

        self.tableau = new_tableau

    def first_non_integer_solution_entry(self):
        """
        Find a entry on solution that is not integer
        :return: Row of b where is a not integer entry
        :rtype: int
        """
        for row in range(self.num_restrictions):
            if self.b_from_tableau[row, 0].denominator is not 1:
                for column in range(self.num_variables):
                    if self.a_from_tableau[row, column] == Fraction(1, 1):
                        return row
        return -1

    def __get_solution_index_from_b(self, row):
        for column in range(self.num_variables):
            if self.a_from_tableau[row, column] == Fraction(1, 1):
                return column

    def cutting_planes(self):
        """
        Do cutting planes in the same instance
        """
        step = 1

        while True:
            row = self.first_non_integer_solution_entry()
            if row is -1:
                break
            self.__print_on_log("-------------------------- CUTTING PLANES " + str(step) + "------------------------\n")
            restriction = np.copy(self.tableau[row + 1:,
                                  self.num_restrictions:self.num_restrictions + self.num_variables_from_tableau + 1])
            for i in range(self.num_variables_from_tableau + 1):
                restriction[0, i] = floor(restriction[0, i])
            self.addRestriction(restriction)
            self.__print_on_log(self.__to_string_tableau() + "\n")
            step += 1

    def __get_new_restriction_bnb(self, value, row, rtype):
        restriction = self.__zeros_matrix(1, self.num_variables_from_tableau + 1)
        if rtype is self.LESS:
            restriction[0, self.basis[row]] = 1
            restriction[0, -1] = value
        elif rtype is self.GREATER:
            restriction[0, self.basis[row]] = -1
            restriction[0, -1] = -value
        return restriction

    def just_print_it_for_me(self):
        self.__print_on_log("Relaxação linear\n")
        self.__print_on_log(self.__to_string_tableau())
        self.__print_on_log(str(self.basis) + "\n\n")

    def branch_and_bound(self, level):
        print("RELAXAÇÃO LINEAR: " + str(self.objective_value) + "\n\n");

        row = self.first_non_integer_solution_entry()
        index = self.__get_solution_index_from_b(row)

        if not self.infeasible and row is -1 and (self.objective_value > self.integer_objective_value or not self.integer_found):
            # self.__print_on_log("Founded integer " + str(self.objective_value) + " " + self.to_string_vector_with_fractions(self.solution) + "\n")
            self.integer_objective_value = self.objective_value
            self.integer_solution = np.copy(self.solution)
            self.integer_found = True
            return
        elif not self.infeasible and (self.integer_objective_value is None or self.objective_value > self.integer_objective_value):
            current_basis = list(self.basis)
            basis_before_new_simplex = list(self.basis)
            basis_before_new_simplex.append(self.a_from_tableau.shape[1])

            # self.__print_on_log("Antes de adicionar a restrição\n")
            # self.__print_on_log(self.__to_string_tableau())
            # self.__print_on_log(str(self.basis) + "\n")

            # Cria nova restrição e adiciona
            restriction = self.__get_new_restriction_bnb(floor(self.b_from_tableau[row, 0]), row, self.LESS)
            # self.__print_on_log("--- ADICIONANDO RESTRIÇÃO " + self.to_string_vector_with_fractions(restriction) + "\n")
            self.addRestriction(restriction)

            # self.__print_on_log("Depois de adicionar a restrição\n")
            # self.__print_on_log(self.__to_string_tableau())
            # self.__print_on_log(str(self.basis) + "\n")

            # Printa no log o estado atual
            self.__print_on_log("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            self.__print_on_log("┃" + ("    " * level) + "[x_" + str(index) + " <= " + str(floor(self.b_from_tableau[row, 0])) + "] ")
            if self.infeasible:
                self.__print_on_log("Infeasible\n")
            else:
                self.__print_on_log(str(self.objective_value) + " " + self.to_string_vector_with_fractions(self.solution))
                if all(self.solution[0, column].denominator == 1 for column in range(self.solution.shape[1])):
                    self.__print_on_log(" Integer\n")
                else:
                    self.__print_on_log("\n")
            self.__print_on_log("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

            # Chama branch and bound para primeira restrição
            self.branch_and_bound(level + 1)
            # self.__print_on_log("Voltou do primeiro B&B\n")

            # Repivoteia no para o estado quando adiciona a nova restrição e não performa simplex
            # self.__print_on_log("Antes de repivotear (preparação para remover restrição)\n")
            # self.__print_on_log(self.__to_string_tableau())
            # self.__print_on_log(str(self.basis) + "\n")
            for i in range(self.num_restrictions):
                self.__pivot_tableau(i, basis_before_new_simplex[i])
            self.basis = list(basis_before_new_simplex)

            # Remove restrição. Base terá um elemento a menos.
            # self.__print_on_log("Antes de remover restrição\n")
            # self.__print_on_log(self.__to_string_tableau())
            # self.__print_on_log(str(self.basis) + "\n")
            self.remove_last_restriction()

            # self.__print_on_log("Antes de adicionar restrição\n")
            # self.__print_on_log(self.__to_string_tableau())
            # self.__print_on_log(str(self.basis) + "\n")

            # Cria a segunda restrição e adiciona. Base voltará a ter um elemento a mais.
            restriction = self.__get_new_restriction_bnb(ceil(self.b_from_tableau[row, 0]), row, self.GREATER)
            # self.__print_on_log("--- ADICIONANDO RESTRIÇÃO " + self.to_string_vector_with_fractions(restriction) + "\n")
            self.addRestriction(restriction)

            # self.__print_on_log("Depois de adicionar a restrição\n")
            # self.__print_on_log(self.__to_string_tableau())
            # self.__print_on_log(str(self.basis) + "\n")

            # Printa no log o estado atual
            self.__print_on_log("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            self.__print_on_log("┃" + ("    " * level) + "[x_" + str(index) + " >= " + str(ceil(self.b_from_tableau[row, 0])) + "] ")
            if self.infeasible:
                self.__print_on_log("Infeasible\n")
            else:
                self.__print_on_log(str(self.objective_value) + " " + self.to_string_vector_with_fractions(self.solution))
                if all(self.solution[0, column].denominator == 1 for column in range(self.solution.shape[1])):
                    self.__print_on_log(" Integer\n")
                else:
                    self.__print_on_log("\n")
            self.__print_on_log("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

            # Chama branch and bound para a segunda restrição
            self.branch_and_bound(level + 1)
            self.__print_on_log("Voltou do segundo B&B com o seguinte tableau e antes de repivotar para remover\n")
            self.__print_on_log(self.__to_string_tableau())
            self.__print_on_log(str(self.basis) + "\n")
            self.__print_on_log(str(basis_before_new_simplex) + "\n")

            # Repivoteia no para o estado quando adiciona a nova restrição e não performa simplex
            for i in range(self.num_restrictions):
                self.__pivot_tableau(i, basis_before_new_simplex[i])
            self.basis = list(basis_before_new_simplex)

            # Remove restrição. Base teŕa um elemento a menos
            self.__print_on_log("Depois de pivotar e antes de remover restrição\n")
            self.__print_on_log(self.__to_string_tableau())
            self.__print_on_log(str(self.basis) + "\n")
            self.remove_last_restriction()

            # self.__print_on_log("Removeu segunda restrição\n")
            # self.__print_on_log(self.__to_string_tableau())
            # self.__print_on_log(str(self.basis) + "\n")
            return
        elif self.infeasible or self.objective_value <= self.integer_objective_value:
            # self.__print_on_log("É inviável\n")
            self.infeasible = False
            return

def main():
    input_file = open(sys.argv[1], "r")
    algorithm = int(input_file.readline().replace("\n", ""))
    lines = int(input_file.readline().replace("\n", ""))
    columns = int(input_file.readline().replace("\n", ""))
    string_matrix = input_file.readline().replace("\n", "")
    input_file.close()
    matrix = json.loads(string_matrix)

    matrix = convert_matrix_to_fractions(matrix)

    log_file = open("log.txt", "w")
    linear_programming: LinearProgramming = LinearProgramming(matrix=matrix, log_file=log_file)

    linear_programming.simplex()

    conclusion_file = open("conclusao.txt", "w")
    if algorithm == 0:
        if linear_programming.infeasible:
            conclusion_file.write("0\n")
        elif linear_programming.unlimited:
            conclusion_file.write("1\n")
            conclusion_file.write(linear_programming.to_string_vector_with_fractions(linear_programming.certificate) + "\n")
        else:
            conclusion_file.write("2\n")
            linear_relaxation_objective_value = linear_programming.objective_value
            linear_relaxation_solution = np.copy(linear_programming.solution)
            linear_relaxation_certificate = np.copy(linear_programming.certificate)
            linear_programming.cutting_planes()
            conclusion_file.write(linear_programming.to_string_vector_with_fractions(linear_programming.solution) + "\n")
            conclusion_file.write(str(linear_programming.objective_value) + "\n")
            conclusion_file.write(linear_programming.to_string_vector_with_fractions(linear_relaxation_solution) + "\n")
            conclusion_file.write(str(linear_relaxation_objective_value) + "\n")
            conclusion_file.write(linear_programming.to_string_vector_with_fractions(linear_relaxation_certificate) + "\n")
    elif algorithm == 1:
        if linear_programming.infeasible:
            conclusion_file.write("0\n")
        elif linear_programming.unlimited:
            conclusion_file.write("1\n")
            conclusion_file.write(linear_programming.to_string_vector_with_fractions(linear_programming.certificate) + "\n")
        else:
            conclusion_file.write("2\n")
            linear_relaxation_objective_value = linear_programming.objective_value
            linear_relaxation_solution = np.copy(linear_programming.solution)
            linear_relaxation_certificate = np.copy(linear_programming.certificate)
            linear_programming.just_print_it_for_me()
            linear_programming.branch_and_bound(0)
            print(linear_programming.integer_objective_value)
            conclusion_file.write(linear_programming.to_string_vector_with_fractions(linear_programming.integer_solution) + "\n")
            conclusion_file.write(str(linear_programming.integer_objective_value) + "\n")
            conclusion_file.write(linear_programming.to_string_vector_with_fractions(linear_relaxation_solution) + "\n")
            conclusion_file.write(str(linear_relaxation_objective_value) + "\n")
            conclusion_file.write(linear_programming.to_string_vector_with_fractions(linear_relaxation_certificate) + "\n")


    conclusion_file.close()

    log_file.close()


if __name__ == "__main__":
    main()
