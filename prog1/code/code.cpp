#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/SparseCholesky> 

using namespace Eigen;

int main(int argc, char* argv[]){

    // STEP 1

    MatrixXd Ag = MatrixXd::Zero(9,9);
    
    Ag.coeffRef(0,1) = 1.0;
    Ag.coeffRef(1,0) = 1.0;
    Ag.coeffRef(0,3) = 1.0;
    Ag.coeffRef(3,0) = 1.0;
    Ag.coeffRef(1,2) = 1.0;
    Ag.coeffRef(2,1) = 1.0;
    Ag.coeffRef(2,3) = 1.0;
    Ag.coeffRef(3,2) = 1.0;
    Ag.coeffRef(2,4) = 1.0;
    Ag.coeffRef(4,2) = 1.0;
    Ag.coeffRef(4,5) = 1.0;
    Ag.coeffRef(5,4) = 1.0;
    Ag.coeffRef(4,7) = 1.0;
    Ag.coeffRef(7,4) = 1.0;
    Ag.coeffRef(4,8) = 1.0;
    Ag.coeffRef(8,4) = 1.0;
    Ag.coeffRef(5,6) = 1.0;
    Ag.coeffRef(6,5) = 1.0;
    Ag.coeffRef(6,8) = 1.0;
    Ag.coeffRef(8,6) = 1.0;
    Ag.coeffRef(6,7) = 1.0;
    Ag.coeffRef(7,6) = 1.0;
    Ag.coeffRef(7,8) = 1.0;
    Ag.coeffRef(8,7) = 1.0;

    // std::cout << "Matrix Ag:\n" << Ag << std::endl;
    std::cout << "The Frobenius norm of Ag is " << Ag.norm() << std::endl;

    // STEP 2

    VectorXd vg(Ag.rows());
    for (int i = 0; i<Ag.rows(); i++){
        int sum = 0;
        for (int j = 0; j<Ag.rows(); j++) {
            sum += Ag(i,j);
        }
        vg[i] = sum;
    }
    
    // std::cout << "Vector vg:\n" << vg << std::endl;

    MatrixXd Dg = vg.asDiagonal();
    //std::cout << "Matrix Dg:\n" << Dg << std::endl;
    MatrixXd Lg = Dg-Ag;
    // std::cout << "Matrix Lg:\n" << Lg << std::endl;

    VectorXd x = VectorXd::Ones(vg.rows());
    MatrixXd y = Lg*x;

    std::cout << "The Euclidean norm of y is " << y.norm() << std::endl;

    SelfAdjointEigenSolver<MatrixXd> saeigensolver(Lg);
    if (saeigensolver.info() != Eigen::Success) abort();
    VectorXd yEigenvalues = saeigensolver.eigenvalues(); 
    if ((yEigenvalues(0) > 1e-10) && (Lg.isApprox(Lg.transpose()))) {
        std::cout << "Lg is Symmetric and Positive Definite" << std::endl;
    }
    else {
        std::cout << "Lg is not Symmetric and Positive Definite" << std::endl;
    }

    // STEP 3

    std::cout << "The smallest eigenvalue is " << yEigenvalues(0) << std::endl;
    std::cout << "The largest eigenvalue is " << yEigenvalues(yEigenvalues.size() - 1) << std::endl;

    // std::cout << "All the eigenvalues are\n" << yEigenvalues << std::endl;
    
    MatrixXd yEigenvectors = saeigensolver.eigenvectors();
    // std::cout << "The matrix of the eigenvectors is\n" << yEigenvectors << std::endl;
    
    // STEP 4

    std::cout << "The smallest strictly positive eigenvalue is " << yEigenvalues(1) << std::endl;
    std::cout << "The corresponding eigenvector is\n" << yEigenvectors.col(1) << std::endl;

    VectorXd fiedlerVector = yEigenvectors.col(1);

    std::cout << "The nodes with positive entries are: { ";
    for (int i = 0; i < fiedlerVector.size(); ++i) {
        if (fiedlerVector(i) > 1e-12) {
            std::cout << (i + 1) << ";";
        }
    }
    std::cout << "}" << std::endl;

    std::cout << "The nodes with negative entries are: { ";
    for (int i = 0; i < fiedlerVector.size(); ++i) {
        if (fiedlerVector(i) < -1e-12) {
            std::cout << (i + 1) << ";";
        }
    }
    std::cout << "}" << std::endl;

    // STEP 5

    SparseMatrix<double> As;
    loadMarket(As, "social.mtx");
    std::cout << "The Frobienius norm of As is " << As.norm() << std::endl;

    // STEP 6

    VectorXd vs = As * VectorXd::Ones(As.cols());
    //std::cout << "Vector vs:\n" << vs << std::endl;

    SparseMatrix<double> Ds(As.rows(), As.cols());
    for (int i = 0; i<vs.size(); i++) {
        Ds.insert(i,i) = vs(i);
    }
    //std::cout << "Matrix Ds:\n" << Ds << std::endl;

    SparseMatrix<double> Ls = Ds-As;
    std::cout << "Is Ls symmetric? " << ((Ls.isApprox(Ls.transpose())) ? "Yes" : "No") << std::endl;
    std::cout << "The non-zero entries of Ls are " << Ls.nonZeros() << std::endl;

    // STEP 7

    // std::cout << Ls(1,1) << std::endl;
    Ls.coeffRef(0,0) = Ls.coeffRef(0,0) + 0.2;
    // std::cout << Ls(1,1) << std::endl;

    if (!Eigen::saveMarket(Ls, "Ls.mtx")) {
        std::cerr << "Error saving Ls.mtx" << std::endl;
    }

    // ../lis-2.1.10/test/eigen1 Ls.mtx eigvec.txt hist.txt -e pi -etol 1.e-8 -emaxiter 5000
    /*
    number of processes = 1
    matrix size = 351 x 351 (9153 nonzero entries)

    initial vector x      : all components set to 1
    precision             : double
    eigensolver           : Power
    convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-08 * ||lx||_2
    matrix storage format : CSR
    shift                 : 0.000000e+00
    eigensolver status    : normal end

    Power: mode number          = 0
    Power: eigenvalue           = 6.013370e+01
    Power: number of iterations = 2007
    Power: elapsed time         = 2.666679e-01 sec.
    Power:   preconditioner     = 0.000000e+00 sec.
    Power:     matrix creation  = 0.000000e+00 sec.
    Power:   linear solver      = 0.000000e+00 sec.
    Power: relative residual    = 9.940435e-09
    */

    // STEP 8

    // ../lis-2.1.10/test/eigen1 Ls.mtx eigvec.txt hist.txt -e pi -etol 1.e-8 -shift 29
    /*
    number of processes = 1
    matrix size = 351 x 351 (9153 nonzero entries)

    initial vector x      : all components set to 1
    precision             : double
    eigensolver           : Power
    convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-08 * ||lx||_2
    matrix storage format : CSR
    shift                 : 2.900000e+01    
    eigensolver status    : LIS_MAXITER(code=4)

    Power: mode number          = 0
    Power: eigenvalue           = 6.013370e+01
    Power: number of iterations = 1000
    Power: elapsed time         = 1.186547e-01 sec.
    Power:   preconditioner     = 0.000000e+00 sec.
    Power:     matrix creation  = 0.000000e+00 sec.
    Power:   linear solver      = 0.000000e+00 sec.
    Power: relative residual    = 3.238162e-08
    
    */
    // ../lis-2.1.10/test/eigen1 Ls.mtx eigvec.txt hist.txt -e ii -etol 1.e-8 -shift 6.013370e+01
    /*
    number of processes = 1
    matrix size = 351 x 351 (9153 nonzero entries)

    initial vector x      : all components set to 1
    precision             : double
    eigensolver           : Inverse
    convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-08 * ||lx||_2
    matrix storage format : CSR 
    shift                 : 6.013370e+01
    linear solver         : BiCG
    preconditioner        : none
    eigensolver status    : normal end

    Inverse: mode number          = 0
    Inverse: eigenvalue           = 6.013370e+01
    Inverse: number of iterations = 3
    Inverse: elapsed time         = 6.539031e-02 sec.
    Inverse:   preconditioner     = 1.460825e-03 sec.
    Inverse:     matrix creation  = 1.494000e-05 sec.
    Inverse:   linear solver      = 6.181583e-02 sec.
    Inverse: relative residual    = 6.274599e-10    
    */

    // STEP 9

    // mpirun -n 2 ../lis-2.1.10/test/eigen2 Ls.mtx evals.mtx eigvecs.mtx res.txt iters.txt -e si -ss 2 -etol 1.0e-10
    /*
    number of processes = 2
    matrix size = 351 x 351 (9153 nonzero entries)

    initial vector x      : all components set to 1
    precision             : double
    eigensolver           : Subspace
    convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-10 * ||lx||_2
    matrix storage format : CSR
    shift                 : 0.000000e+00
    inner eigensolver     : Inverse
    linear solver         : BiCG
    preconditioner        : none
    size of subspace      : 2

    compute eigenpairs in subspace:

    Subspace: mode number          = 0
    Subspace: eigenvalue           = 5.669404e-04
    Subspace: elapsed time         = 2.796012e-02 sec.
    Subspace: number of iterations = 3
    Subspace: relative residual    = 4.512722e-12

    Subspace: mode number          = 1
    Subspace: eigenvalue           = 1.789070e+00
    Subspace: elapsed time         = 3.808057e-01 sec.
    Subspace: number of iterations = 113
    Subspace: relative residual    = 8.965728e-11

    eigensolver status    : normal end

    Subspace: mode number          = 0
    Subspace: eigenvalue           = 5.669404e-04
    Subspace: number of iterations = 3
    Subspace: elapsed time         = 4.106167e-01 sec.
    Subspace:   preconditioner     = 4.446000e-04 sec.
    Subspace:     matrix creation  = 1.296000e-05 sec.
    Subspace:   linear solver      = 2.665827e-02 sec.
    Subspace: relative residual    = 4.512722e-12
    */

    // STEP 10

    SparseMatrix<double> eigvecs;
    Eigen::loadMarket(eigvecs, "eigvecs.mtx");
    
    VectorXd eigvec2 = eigvecs.col(1);
    
    int np = 0;
    int nn = 0;

    std::vector<int> positive_indices;
    std::vector<int> negative_indices;

    for (int i = 0; i < eigvec2.size(); ++i) {
        if (eigvec2(i) >= 0) {
            np++;
            positive_indices.push_back(i);
        } else {
            nn++;
            negative_indices.push_back(i);
        }
    }

    std::cout << "The number of positive entries is " << np << std::endl;
    std::cout << "The number of negative entries is " << nn << std::endl;

    // STEP 11

    VectorXi eigen_indices(np + nn);
    for(int i = 0; i < np; ++i) {
        eigen_indices(i) = positive_indices[i];
    }
    for(int i = 0; i < nn; ++i) {
        eigen_indices(np + i) = negative_indices[i];
    }

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P(eigen_indices); // this is the transpose of the real P matrix

    SparseMatrix<double> Aord = P.transpose()*As*P;

    SparseMatrix<double> offDiagonalBlockAord = Aord.block(0, np, np, nn);
    int nonZerosAord = offDiagonalBlockAord.nonZeros();
    std::cout << "The non-zero entries of Aord " << nonZerosAord << std::endl;

    SparseMatrix<double> offDiagonalBlockAs = As.block(0,np, np, nn);
    int nonZerosAs = offDiagonalBlockAs.nonZeros();
    std::cout << "The non-zero entries of As " << nonZerosAs << std::endl;

    SparseMatrix<double> offDiagonalBlockAord2 = Aord.block(np, 0, nn, np);
    int nonZerosAord2 = offDiagonalBlockAord2.nonZeros();
    std::cout << "The non-zero entries of Aord " << nonZerosAord2 << std::endl;

    SparseMatrix<double> offDiagonalBlockAs2 = As.block(np, 0 , nn, np);
    int nonZerosAs2 = offDiagonalBlockAs2.nonZeros();
    std::cout << "The non-zero entries of As " << nonZerosAs2 << std::endl;

    return 0;
}
