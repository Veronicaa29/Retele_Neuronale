import pathlib
import math
import copy

def load_system(path : pathlib.Path) -> tuple[list[list[float]], list[float]]:
    A = []
    B = []

    with path.open() as file:
        for line in file:
            line = line.strip()
            equation, constant = line.split('=')
            # print("Equation:", equation)

            equation = "b" + equation
            equation_terms = equation.replace(' ', '').replace('x', ' ').replace('y', ' ').replace('z', ' ').split()
            # print("Equation terms:", equation_terms)
            new_equation_terms = []
            for term in equation_terms:
                if len(term) == 1 and term == 'b':
                    term = '1'
                if len(term) > 1 and term[0] == 'b':
                    term = term[1:]
                if term == '+':
                    term = '1'
                if term == '-':
                    term = term + '1'
                if term[0] == '+':
                    term = term[1:]
                new_equation_terms.append(float(term))

            # print("New equation terms:", new_equation_terms)

            constant = float(constant.strip()) # elimin spatiile de la inceput si sfarsit

            A.append(new_equation_terms)
            B.append(constant)

    return A, B

def determinant(matrix: list[list[float]]) -> float:
    if len(matrix) == 1:
        return matrix[0][0]
    elif len(matrix) == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    else:
        return matrix[0][0]*(matrix[1][1]*matrix[2][2] - matrix[1][2]*matrix[2][1]) -matrix[0][1]*(matrix[1][0]*matrix[2][2] - matrix[1][2]*matrix[2][0]) + matrix[0][2]*(matrix[1][0]*matrix[2][1] - matrix[1][1]*matrix[2][0])

def trace(matrix: list[list[float]]) -> float:
    return matrix[0][0] + matrix[1][1] + matrix[2][2]

def norm(vector: list[float]) -> float:
    return math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)

def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    result = [0, 0, 0]
    for i in range(len(matrix)):
        for j in range(len(vector)):
            result[i] += matrix[i][j] * vector[j]

    return result

def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det_A = determinant(matrix)

    x_matrix = copy.deepcopy(matrix)
    for i in range(len(matrix)):
        x_matrix[i][0] = vector[i]
    x = determinant(x_matrix) / det_A

    y_matrix = copy.deepcopy(matrix)
    for i in range(len(matrix)):
        x_matrix[i][1] = vector[i]
    y = determinant(y_matrix) / det_A

    z_matrix = copy.deepcopy(matrix)
    for i in range(len(matrix)):
        x_matrix[i][2] = vector[i]
    z = determinant(z_matrix) / det_A

    return [x, y, z]

def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    minor_matrix = []

    for a in range(len(matrix)):
        line = []
        for b in range(len(matrix[a])):
            if a != i and b != j:
                line.append(matrix[a][b])
        if line:
            minor_matrix.append(line)

    return minor_matrix

def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    cofactor_matrix = []

    for i in range(len(matrix)):
        line = []
        for j in range(len(matrix[i])):
            minor_matrix = minor(matrix, i, j)
            line.append(((-1)**(i+j)) * determinant(minor_matrix))
        cofactor_matrix.append(line)

    return cofactor_matrix

def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(cofactor(matrix))

def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    adjoint_matrix = adjoint(matrix)
    det_A = determinant(matrix)

    result = multiply(adjoint_matrix, vector)

    for i in range(len(result)):
        result[i] /= det_A

    return result


def main():
    path = pathlib.Path("InputFile.txt")
    A, B = load_system(path)

    print("A=", A)
    print("B=", B)
    print(f"{determinant(A)=}")
    print(f"{trace(A)=}")
    print(f"{norm(B)=}")
    print(f"{transpose(A)=}")
    print(f"{multiply(A, B)=}")
    print(f"{solve_cramer(A, B)=}")
    print(f"{minor(A, 1, 1)=}")
    print(f"{cofactor(A)=}")
    print(f"{solve(A, B)=}")

if __name__ == "__main__":
    main()