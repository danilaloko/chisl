#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

constexpr double kTiny = 1e-12;

double f(const Vector& x) {
    // Replace this function with the target function from your assignment.
    double result = 0.0;
    for (double value : x) {
        result += value * value;
    }
    return result;
}

Vector gradAnalytic(const Vector& x) {
    Vector result(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        result[i] = 2.0 * x[i];
    }
    return result;
}

Vector gradNumeric(const Vector& x, double h = 1e-6) {
    Vector result(x.size(), 0.0);

    for (std::size_t i = 0; i < x.size(); ++i) {
        Vector xForward = x;
        Vector xBackward = x;
        xForward[i] += h;
        xBackward[i] -= h;
        result[i] = (f(xForward) - f(xBackward)) / (2.0 * h);
    }

    return result;
}

double dot(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes do not match.");
    }

    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

double norm(const Vector& v) {
    return std::sqrt(dot(v, v));
}

Vector add(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes do not match.");
    }

    Vector result(a.size(), 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

Vector subtract(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes do not match.");
    }

    Vector result(a.size(), 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

Vector multiply(double scalar, const Vector& v) {
    Vector result(v.size(), 0.0);
    for (std::size_t i = 0; i < v.size(); ++i) {
        result[i] = scalar * v[i];
    }
    return result;
}

Matrix identityMatrix(std::size_t n) {
    Matrix result(n, Vector(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        result[i][i] = 1.0;
    }
    return result;
}

Vector multiply(const Matrix& matrix, const Vector& vector) {
    if (matrix.size() != vector.size()) {
        throw std::invalid_argument("Matrix and vector sizes do not match.");
    }

    Vector result(vector.size(), 0.0);
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        if (matrix[i].size() != vector.size()) {
            throw std::invalid_argument("Matrix must be square.");
        }
        for (std::size_t j = 0; j < vector.size(); ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    return result;
}

Matrix outerProduct(const Vector& a, const Vector& b) {
    Matrix result(a.size(), Vector(b.size(), 0.0));
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < b.size(); ++j) {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}

Matrix add(const Matrix& a, const Matrix& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Matrix sizes do not match.");
    }

    Matrix result = a;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) {
            throw std::invalid_argument("Matrix sizes do not match.");
        }
        for (std::size_t j = 0; j < a[i].size(); ++j) {
            result[i][j] += b[i][j];
        }
    }
    return result;
}

Matrix subtract(const Matrix& a, const Matrix& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Matrix sizes do not match.");
    }

    Matrix result = a;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) {
            throw std::invalid_argument("Matrix sizes do not match.");
        }
        for (std::size_t j = 0; j < a[i].size(); ++j) {
            result[i][j] -= b[i][j];
        }
    }
    return result;
}

Matrix divide(const Matrix& matrix, double scalar) {
    if (std::abs(scalar) < kTiny) {
        throw std::runtime_error("Division by a near-zero value.");
    }

    Matrix result = matrix;
    for (Vector& row : result) {
        for (double& value : row) {
            value /= scalar;
        }
    }
    return result;
}

double lineSearchGoldenSection(const Vector& x, const Vector& p) {
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
    const double tolerance = 1e-8;
    const int maxExpandIterations = 60;
    const int maxIterations = 120;

    auto oneDimensionalFunction = [&](double lambda) {
        return f(add(x, multiply(lambda, p)));
    };

    double left = 0.0;
    double right = 1.0;
    double fLeft = oneDimensionalFunction(left);
    double fRight = oneDimensionalFunction(right);

    int expandIteration = 0;
    while (fRight < fLeft && expandIteration < maxExpandIterations) {
        left = right;
        fLeft = fRight;
        right *= 2.0;
        fRight = oneDimensionalFunction(right);
        ++expandIteration;
    }

    double a = 0.0;
    double b = right;
    double c = b - (b - a) / phi;
    double d = a + (b - a) / phi;
    double fC = oneDimensionalFunction(c);
    double fD = oneDimensionalFunction(d);

    for (int iteration = 0; iteration < maxIterations && std::abs(b - a) > tolerance; ++iteration) {
        if (fC < fD) {
            b = d;
            d = c;
            fD = fC;
            c = b - (b - a) / phi;
            fC = oneDimensionalFunction(c);
        } else {
            a = c;
            c = d;
            fC = fD;
            d = a + (b - a) / phi;
            fD = oneDimensionalFunction(d);
        }
    }

    return (a + b) / 2.0;
}

Matrix updateDfpMatrix(const Matrix& h, const Vector& s, const Vector& y) {
    const double sTy = dot(s, y);
    const Vector hY = multiply(h, y);
    const double yThY = dot(y, hY);

    if (std::abs(sTy) < kTiny || std::abs(yThY) < kTiny) {
        return identityMatrix(h.size());
    }

    const Matrix firstCorrection = divide(outerProduct(s, s), sTy);
    const Matrix secondCorrection = divide(outerProduct(hY, hY), yThY);
    return subtract(add(h, firstCorrection), secondCorrection);
}

void printVector(const std::string& name, const Vector& vector) {
    std::cout << name << " = (";
    for (std::size_t i = 0; i < vector.size(); ++i) {
        std::cout << vector[i];
        if (i + 1 != vector.size()) {
            std::cout << ", ";
        }
    }
    std::cout << ")";
}

int main() {
    try {
        std::size_t n = 0;
        double epsilon = 0.0;
        int maxIterations = 0;
        int gradientMode = 0;

        std::cout << "n: ";
        std::cin >> n;
        if (n == 0) {
            throw std::invalid_argument("n must be positive.");
        }

        Vector x(n, 0.0);
        std::cout << "Initial point x0 (" << n << " values): ";
        for (double& value : x) {
            std::cin >> value;
        }

        std::cout << "epsilon: ";
        std::cin >> epsilon;
        if (epsilon <= 0.0) {
            throw std::invalid_argument("epsilon must be positive.");
        }

        std::cout << "max iterations: ";
        std::cin >> maxIterations;
        if (maxIterations <= 0) {
            throw std::invalid_argument("max iterations must be positive.");
        }

        std::cout << "Gradient mode (1 - analytic, 2 - numeric): ";
        std::cin >> gradientMode;
        if (gradientMode != 1 && gradientMode != 2) {
            throw std::invalid_argument("Gradient mode must be 1 or 2.");
        }

        std::function<Vector(const Vector&)> grad;
        if (gradientMode == 1) {
            grad = [](const Vector& point) {
                return gradAnalytic(point);
            };
        } else {
            grad = [](const Vector& point) {
                return gradNumeric(point);
            };
        }
        Matrix h = identityMatrix(n);

        std::cout << std::fixed << std::setprecision(10);
        std::cout << "\niter\tf(x)\t\t||grad||\tlambda\t\tx\n";

        int iteration = 0;
        for (; iteration < maxIterations; ++iteration) {
            const Vector g = grad(x);
            const double gradientNorm = norm(g);

            std::cout << iteration << '\t' << f(x) << '\t' << gradientNorm << '\t';
            if (gradientNorm < epsilon) {
                std::cout << "-\t\t";
                printVector("", x);
                std::cout << "\n";
                break;
            }

            Vector p = multiply(-1.0, multiply(h, g));
            if (dot(p, g) >= 0.0) {
                h = identityMatrix(n);
                p = multiply(-1.0, g);
            }

            const double lambda = lineSearchGoldenSection(x, p);
            const Vector xNext = add(x, multiply(lambda, p));
            const Vector gNext = grad(xNext);
            const Vector s = subtract(xNext, x);
            const Vector y = subtract(gNext, g);

            h = updateDfpMatrix(h, s, y);

            std::cout << lambda << '\t';
            printVector("", x);
            std::cout << "\n";

            x = xNext;
        }

        const Vector finalGradient = grad(x);
        std::cout << "\nResult:\n";
        printVector("x*", x);
        std::cout << "\nf(x*) = " << f(x) << "\n";
        std::cout << "||grad(x*)|| = " << norm(finalGradient) << "\n";
        std::cout << "iterations = " << iteration << "\n";
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << "\n";
        return 1;
    }

    return 0;
}
