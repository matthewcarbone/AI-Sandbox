#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// ---------- Data Structures ----------

struct Primitive {
    double exponent;
    double coefficient;
    double norm; // Normalization factor for an s–type Gaussian: (2α/π)^(3/4)
};

struct BasisFunction {
    string element;           // Element symbol (e.g. "H")
    Vector3d center;          // Center (in bohr)
    vector<Primitive> primitives; // Contracted basis: a list of primitives
};

struct Atom {
    string element;   // Element symbol
    Vector3d position; // Coordinates (in bohr)
    int Z;           // Atomic number
};

// A simple atomic number lookup (only H supported in this example)
int atomic_number(const string &elem) {
    if(elem == "H") return 1;
    // Extend for other elements as needed...
    return 0;
}

// ---------- Integral Functions ----------

// Boys function for n = 0 (F0). For small t the limit is 1.
double boys_function(double t) {
    if (t < 1e-8)
        return 1.0;
    return 0.5 * sqrt(M_PI / t) * erf(sqrt(t));
}

// Overlap integral between two s–type Gaussian primitives.
double gaussian_overlap(double a, const Vector3d &A, double b, const Vector3d &B) {
    double rab2 = (A - B).squaredNorm();
    double pre = pow(M_PI / (a + b), 1.5);
    return pre * exp(-a * b / (a + b) * rab2);
}

// Kinetic energy integral between two s–type Gaussian primitives.
double gaussian_kinetic(double a, const Vector3d &A, double b, const Vector3d &B) {
    double rab2 = (A - B).squaredNorm();
    double pre = a * b / (a + b);
    return pre * (3 - 2 * pre * rab2) * gaussian_overlap(a, A, b, B);
}

// Nuclear attraction integral for a single nucleus located at C.
double gaussian_nuclear(double a, const Vector3d &A,
                        double b, const Vector3d &B,
                        const Vector3d &C) {
    double p = a + b;
    Vector3d P = (a * A + b * B) / p;
    double rpc2 = (P - C).squaredNorm();
    double F0 = boys_function(p * rpc2);
    double pre = 2.0 * M_PI / p;
    return pre * exp(-a * b / p * (A - B).squaredNorm()) * F0;
}

// Two–electron repulsion integral between four s–type Gaussian primitives.
double gaussian_eri(double a, const Vector3d &A,
                    double b, const Vector3d &B,
                    double c, const Vector3d &C,
                    double d, const Vector3d &D) {
    double p = a + b;
    double q = c + d;
    Vector3d P = (a * A + b * B) / p;
    Vector3d Q = (c * C + d * D) / q;
    double rab2 = (A - B).squaredNorm();
    double rcd2 = (C - D).squaredNorm();
    double rpq2 = (P - Q).squaredNorm();
    double pre = 2 * pow(M_PI, 2.5) / (p * q * sqrt(p + q));
    return pre * exp(-a * b / p * rab2 - c * d / q * rcd2) * boys_function((p * q / (p + q)) * rpq2);
}

// ---------- Contracted Basis Function Integrals ----------

double basis_overlap(const BasisFunction &bf1, const BasisFunction &bf2) {
    double sum = 0.0;
    for (const auto &p : bf1.primitives) {
        for (const auto &q : bf2.primitives) {
            sum += p.coefficient * q.coefficient * p.norm * q.norm *
                   gaussian_overlap(p.exponent, bf1.center, q.exponent, bf2.center);
        }
    }
    return sum;
}

double basis_kinetic(const BasisFunction &bf1, const BasisFunction &bf2) {
    double sum = 0.0;
    for (const auto &p : bf1.primitives) {
        for (const auto &q : bf2.primitives) {
            sum += p.coefficient * q.coefficient * p.norm * q.norm *
                   gaussian_kinetic(p.exponent, bf1.center, q.exponent, bf2.center);
        }
    }
    return sum;
}

// Sum over all nuclei: V = ∑₍A₎ –Z_A ⟨φ_i|1/|r–R_A||φ_j⟩.
double basis_nuclear(const BasisFunction &bf1, const BasisFunction &bf2,
                     const Vector3d &nuc_pos, double Z) {
    double sum = 0.0;
    for (const auto &p : bf1.primitives) {
        for (const auto &q : bf2.primitives) {
            sum += p.coefficient * q.coefficient * p.norm * q.norm *
                   gaussian_nuclear(p.exponent, bf1.center, q.exponent, bf2.center, nuc_pos);
        }
    }
    return -Z * sum;
}

double basis_eri(const BasisFunction &bf1, const BasisFunction &bf2,
                 const BasisFunction &bf3, const BasisFunction &bf4) {
    double sum = 0.0;
    for (const auto &p : bf1.primitives)
        for (const auto &q : bf2.primitives)
            for (const auto &r : bf3.primitives)
                for (const auto &s : bf4.primitives)
                    sum += p.coefficient * q.coefficient * r.coefficient * s.coefficient *
                           p.norm * q.norm * r.norm * s.norm *
                           gaussian_eri(p.exponent, bf1.center, q.exponent, bf2.center,
                                        r.exponent, bf3.center, s.exponent, bf4.center);
    return sum;
}

// Helper for indexing a 4D array stored as 1D.
inline int eri_index(int i, int j, int k, int l, int n) {
    return i * n * n * n + j * n * n + k * n + l;
}

// ---------- Main Program: SCF for H₂ ----------

int main(int argc, char** argv) {
    // Start a timer.
    auto start = chrono::high_resolution_clock::now();

    // Define the molecule: H₂ (coordinates in bohr)
    // For H₂ a bond length of ~1.4 bohr corresponds to about 0.74 Å.
    vector<Atom> atoms;
    {
        Atom H;
        H.element = "H";
        H.position = Vector3d(0.0, 0.0, -0.7);
        H.Z = atomic_number("H");
        atoms.push_back(H);
    }
    {
        Atom H;
        H.element = "H";
        H.position = Vector3d(0.0, 0.0, 0.7);
        H.Z = atomic_number("H");
        atoms.push_back(H);
    }

    // Build basis functions from atoms using the STO-3G parameters for hydrogen.
    vector<BasisFunction> basis;
    // STO-3G parameters for hydrogen (exponents and contraction coefficients)
    vector<double> H_exponents = {3.42525091, 0.62391373, 0.16885540};
    vector<double> H_coeffs    = {0.15432897, 0.53532814, 0.44463454};
    for (const auto &atom : atoms) {
        BasisFunction bf;
        bf.element = atom.element;
        bf.center = atom.position;
        for (int i = 0; i < 3; i++) {
            Primitive prim;
            prim.exponent = H_exponents[i];
            prim.coefficient = H_coeffs[i];
            prim.norm = pow(2 * prim.exponent / M_PI, 0.75); // Normalization factor for an s-orbital
            bf.primitives.push_back(prim);
        }
        basis.push_back(bf);
    }
    int nbasis = basis.size();

    // Compute the nuclear repulsion energy: E_nuc = ∑₍i<j₎ Z_i Z_j / R_ij.
    double Enuc = 0.0;
    for (size_t i = 0; i < atoms.size(); i++) {
        for (size_t j = i + 1; j < atoms.size(); j++) {
            double R = (atoms[i].position - atoms[j].position).norm();
            Enuc += atoms[i].Z * atoms[j].Z / R;
        }
    }

    // Build one–electron integrals: Overlap (S), kinetic (T), and nuclear attraction (V).
    MatrixXd S = MatrixXd::Zero(nbasis, nbasis);
    MatrixXd Tmat = MatrixXd::Zero(nbasis, nbasis);
    MatrixXd Vmat = MatrixXd::Zero(nbasis, nbasis);
    MatrixXd Hcore = MatrixXd::Zero(nbasis, nbasis);
    for (int i = 0; i < nbasis; i++) {
        for (int j = 0; j < nbasis; j++) {
            S(i, j) = basis_overlap(basis[i], basis[j]);
            Tmat(i, j) = basis_kinetic(basis[i], basis[j]);
            double Vsum = 0.0;
            for (const auto &atom : atoms) {
                Vsum += basis_nuclear(basis[i], basis[j], atom.position, atom.Z);
            }
            Vmat(i, j) = Vsum;
            Hcore(i, j) = Tmat(i, j) + Vmat(i, j);
        }
    }

    // Build the two–electron repulsion integrals (ERI).
    // Store them in a flattened 1D vector using the index: (i,j,k,l) -> i*n^3 + j*n^2 + k*n + l.
    vector<double> eri(nbasis * nbasis * nbasis * nbasis, 0.0);
    for (int i = 0; i < nbasis; i++)
        for (int j = 0; j < nbasis; j++)
            for (int k = 0; k < nbasis; k++)
                for (int l = 0; l < nbasis; l++) {
                    int index = eri_index(i, j, k, l, nbasis);
                    eri[index] = basis_eri(basis[i], basis[j], basis[k], basis[l]);
                }

    // Determine the total number of electrons and the number of doubly–occupied orbitals.
    int nelectrons = 0;
    for (const auto &atom : atoms)
        nelectrons += atom.Z;
    int ndocc = nelectrons / 2;  // closed–shell system

    // Initialize the density matrix (P) to zero.
    MatrixXd P = MatrixXd::Zero(nbasis, nbasis);

    double E_old = 0.0;
    double convergence_threshold = 1e-6;
    int max_iter = 100;
    double E_total = 0.0;

    // SCF loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Build the Fock matrix: F = H_core + G(P)
        MatrixXd F = Hcore;
        for (int i = 0; i < nbasis; i++) {
            for (int j = 0; j < nbasis; j++) {
                double sum = 0.0;
                for (int k = 0; k < nbasis; k++) {
                    for (int l = 0; l < nbasis; l++) {
                        int ijkl = eri_index(i, j, k, l, nbasis);
                        int ikjl = eri_index(i, k, j, l, nbasis);
                        sum += P(k, l) * (eri[ijkl] - 0.5 * eri[ikjl]);
                    }
                }
                F(i, j) += sum;
            }
        }

        // Solve the generalized eigenvalue problem F C = S C ε.
        // First, form S^(-1/2) by diagonalizing S.
        SelfAdjointEigenSolver<MatrixXd> es(S);
        MatrixXd S_diag = es.eigenvalues().asDiagonal();
        MatrixXd S_evec = es.eigenvectors();
        // Compute S^(-1/2) = U * diag(1/sqrt(s)) * Uᵀ.
        MatrixXd S_inv_sqrt = S_evec * S_diag.diagonal().cwiseInverse().cwiseSqrt().asDiagonal() * S_evec.transpose();

        MatrixXd F_prime = S_inv_sqrt.transpose() * F * S_inv_sqrt;
        SelfAdjointEigenSolver<MatrixXd> esF(F_prime);
        MatrixXd C_prime = esF.eigenvectors();
        MatrixXd C = S_inv_sqrt * C_prime;

        // Form the new density matrix: P(μ,ν) = 2 ∑₍m=1₎^(ndocc) C(μ,m) C(ν,m)
        MatrixXd P_new = MatrixXd::Zero(nbasis, nbasis);
        for (int mu = 0; mu < nbasis; mu++) {
            for (int nu = 0; nu < nbasis; nu++) {
                double sum = 0.0;
                for (int m = 0; m < ndocc; m++)
                    sum += 2 * C(mu, m) * C(nu, m);
                P_new(mu, nu) = sum;
            }
        }

        // Compute the electronic energy:
        double E_elec = 0.0;
        for (int i = 0; i < nbasis; i++)
            for (int j = 0; j < nbasis; j++)
                E_elec += 0.5 * P_new(i, j) * (Hcore(i, j) + F(i, j));
        E_total = E_elec + Enuc;

        cout << "Iteration " << iter + 1 << ": Energy = " << E_total << " Hartree" << endl;
        if (fabs(E_total - E_old) < convergence_threshold)
            break;
        E_old = E_total;
        P = P_new;
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "\nFinal SCF Energy: " << E_total << " Hartree" << endl;
    cout << "Nuclear Repulsion Energy: " << Enuc << " Hartree" << endl;
    cout << "Elapsed time: " << elapsed.count() << " seconds" << endl;
    cout << "Expected energy (approx.): -1.117 Hartree" << endl;

    return 0;
}
