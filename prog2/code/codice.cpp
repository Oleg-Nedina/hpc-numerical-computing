#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/IterativeLinearSolvers>


#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdlib>
#include <ctime>

using namespace Eigen;

MatrixXd convertToGrayscale(const MatrixXd& red, const MatrixXd& green,
                            const MatrixXd& blue) {
  return 0.299 * red + 0.587 * green + 0.114 * blue;

}

MatrixXd addNoise(const MatrixXd& M) {
    int h = M.rows();
    int w = M.cols();


    MatrixXd noise2 = MatrixXd::Random(h , w)*(40.0/255.0);
    MatrixXd noise = noise2 + M;
    for(int i=0 ; i<h ; ++i){
        for(int j = 0 ; j < w ; ++j){

            if(noise(i,j) < 0) noise(i,j) = 0;
            if (noise(i,j) > 1) noise(i,j) = 1;

        }
    }
    return noise;
}

SparseMatrix<double> buildConvolutionMatrix(int height, int width) {
    typedef Eigen::Triplet<double> T;
    std::vector<T> triplets;

    // Kernel Hav1
    double kernel[3][3] = {
        {1.0/8, 1.0/8, 0.0},
        {1.0/8, 2.0/8, 1.0/8},
        {0.0,   1.0/8, 1.0/8}
    };

    // Loop over every pixel
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int row = i * width + j; // posizione nel vettore

            // Apply 3x3 kernel around (i,j)
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    int ni = i + ki;
                    int nj = j + kj;
                    if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        int col = ni * width + nj;
                        double val = kernel[ki+1][kj+1];
                        if (val != 0.0) {
                            triplets.push_back(T(row, col, val));
                        }
                    }
                }
            }
        }
    }

    SparseMatrix<double> A1(height*width, height*width);
    A1.setFromTriplets(triplets.begin(), triplets.end());
    return A1;
}

SparseMatrix<double> buildSharpenMatrix(int height, int width) {
    typedef Eigen::Triplet<double> T;
    std::vector<T> triplets;

    double kernel[3][3] = {
        {0, -2, 0},
        {-2, 9, -2},
        {0, -2, 0}
    };

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int row = i * width + j;

            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    int ni = i + ki;
                    int nj = j + kj;
                    if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        int col = ni * width + nj;
                        double val = kernel[ki+1][kj+1];
                        if (val != 0.0) {
                            triplets.push_back(T(row, col, val));
                        }
                    }
                }
            }
        }
    }

    SparseMatrix<double> A2(height*width, height*width);
    A2.setFromTriplets(triplets.begin(), triplets.end());
    return A2;
}


SparseMatrix<double> buildEdgeMatrix(int height, int width) {
    typedef Eigen::Triplet<double> T;
    std::vector<T> triplets;

    double kernel[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int row = i * width + j;
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    int ni = i + ki;
                    int nj = j + kj;
                    if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        int col = ni * width + nj;
                        double val = kernel[ki+1][kj+1];
                        if(val != 0.0) {
                            triplets.push_back(T(row, col, val));
                        }
                    }
                }
            }
        }
    }

    SparseMatrix<double> A3(height*width, height*width);
    A3.setFromTriplets(triplets.begin(), triplets.end());
    return A3;
}

bool loadLisVector(Eigen::VectorXd& vec, const std::string& filename) {
    std::ifstream file(filename);
    std::string line;

    std::getline(file, line);
    int vector_size = 0;

    if (file >> vector_size) {
        vec.resize(vector_size);
        vec.setZero();
    } else {
        std::cerr << "Errore: impossibile leggere la dimensione del vettore." << std::endl;
        return false;
    }

    std::getline(file, line);

    int index;
    double value;
    while (file >> index >> value) {
        if (index > 0 && index <= vector_size) {
            vec(index - 1) = value;
        }
    }

    file.close();
    return true;
}
auto normalize_to_uchar = [](const Eigen::VectorXd& vec, int height, int width) {
    double minVal = vec.minCoeff();
    double maxVal = vec.maxCoeff();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat =
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(vec.data(), height, width);

    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> img(height, width);
    img = mat.unaryExpr([=](double v) -> unsigned char {
        double norm = (v - minVal) / (maxVal - minVal + 1e-12); // evita divisione per 0
        return static_cast<unsigned char>(norm * 255.0);
    });
    return img;
};

int main(int argc, char* argv[]){

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <uma.jpg>" << std::endl;
    return 1;
  }

  // STEP 1 

  const char* input_image_path = argv[1];

  int width, height, channels;
  unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 3);  // Force load as RGB
  int mn = width * height ;

 if (!image_data) {
    std::cerr << "Error: Could not load image " << input_image_path << std::endl;
    return 1;
  }

 std::cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << std::endl;

  // Prepare Eigen matrices for each RGB channel
  MatrixXd red(height, width), green(height, width), blue(height, width);

  // Fill the matrices with image data
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int index = (i * width + j) * 3;  // 3 channels (RGB)
      red(i, j) = static_cast<double>(image_data[index]) / 255.0;
      green(i, j) = static_cast<double>(image_data[index + 1]) / 255.0;
      blue(i, j) = static_cast<double>(image_data[index + 2]) / 255.0;
    }
  }
  // Free memory!!!
  stbi_image_free(image_data);

  // Create a grayscale matrix
  MatrixXd gray = convertToGrayscale(red, green, blue);


  // STEP 2

  MatrixXd noiseImage = addNoise(gray);

  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> NOISE(height, width);
  // Use Eigen's unaryExpr to map the grayscale values (0.0 to 1.0) to 0 to 255
  NOISE = noiseImage.unaryExpr([](double val) -> unsigned char {
          return static_cast<unsigned char>(val*255.0);
  });

  Matrix<unsigned char , Dynamic , Dynamic , RowMajor> gray2(height , width);
  gray2 = gray.unaryExpr([](double val) -> unsigned char {
            return static_cast<unsigned char>(val*255.0);
    });


   const std::string output_image_path = "Immagine_Con_Rumore.png";

   std::cout << "noise image saved to " << output_image_path << std::endl;

   if (stbi_write_png(output_image_path.c_str(), width, height, 1,NOISE.data(), width) == 0) {
       std::cerr << "Error: Could not save grayscale image" << std::endl;
       return 1;
   }

   //  STEP 3

   VectorXd v  = VectorXd{gray2.cast<double>().transpose().reshaped()};
   VectorXd w =  VectorXd{NOISE.cast<double>().transpose().reshaped()};

   std::cout << "v has " << v.size() << " components (expected "<< height * width << ")" << std::endl;
   std::cout << "w has " << w.size() << " components (expected " << height * width << ")" << std::endl;

   double norm_v = v.norm();
   std::cout << "Euclidean norm of v = " << norm_v << std::endl;

   // STEP 4

   SparseMatrix<double> A1 = buildConvolutionMatrix(height, width);
   std::cout << "A1 has " << A1.nonZeros() << " non-zero entries." << std::endl;

   // STEP 5

   Eigen::VectorXd g_smooth = A1 * w;
   Matrix<double, Dynamic, Dynamic, RowMajor> smoothImage =  Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(g_smooth.data(), height, width);
   Matrix<unsigned char, Dynamic, Dynamic, RowMajor> smoothImageUC(height, width);
   smoothImageUC = smoothImage.unaryExpr([](double val) -> unsigned char {
           double c = std::min(std::max(val, 0.0), 255.0);
           return static_cast<unsigned char>(c);
           });
   const std::string output_image2 = "Immagine_con_smooth.png";
   if (stbi_write_png(output_image2.c_str(), width, height, 1,smoothImageUC.data(), width) == 0) {
       std::cerr << "Error: Could not save grayscale image" << std::endl;
       return 1;
   }

   // STEP 6
   
    SparseMatrix<double> A2 =  buildSharpenMatrix(height, width);
    std::cout << "A2 has " << A2.nonZeros() << " non-zero entries." << std::endl;
    std::cout << "Is A2 symmetric? " << (A2.isApprox(A2.transpose()) ? "Yes" : "No") << std::endl;

    // STEP 7

    Eigen::VectorXd g_sharp = A2 * v;
    Matrix<double, Dynamic, Dynamic, RowMajor> sharpImage =  Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(g_sharp.data(), height, width);
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> sharpImageUC(height, width);
    sharpImageUC = sharpImage.unaryExpr([](double val) -> unsigned char {
            double c = std::min(std::max(val, 0.0), 255.0);
            return static_cast<unsigned char>(c);
            });
   const std::string output_image3 = "Immagine_con_sharp.png";
   if (stbi_write_png(output_image3.c_str(), width, height, 1,sharpImageUC.data(), width) == 0) {
        std::cerr << "Error: Could not save grayscale image" << std::endl;
        return 1;
    }

   // STEP 8
   
   if (!Eigen::saveMarket(A2, "A2.mtx")) {
        std::cerr << "Error saving A2.mtx" << std::endl;
   }
   int n = w.size();
   Eigen::saveMarketVector(w, "w2.mtx");
   FILE* out = fopen("w2.mtx","w");
   fprintf(out,"%%%%MatrixMarket vector coordinate real general\n");
   fprintf(out,"%d\n",n);
   for(int i = 0 ; i<n ; ++i){
       fprintf(out,"%d %f\n",i,w(i));
   }
   fclose(out);

   // STEP 9

    Eigen::VectorXd x;
    if (!loadLisVector(x, "x.mtx")) {
        return 1;
                  }

        Matrix<double, Dynamic, Dynamic, RowMajor> xImage =  Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(x.data(), height, width);
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> xImageUC(height, width);
    xImageUC = xImage.unaryExpr([](double val) -> unsigned char {
            double c = std::min(std::max(val, 0.0), 255.0);
            return static_cast<unsigned char>(c);
            });
    const std::string output_image4 = "Immagine_LIS.png";
    if (stbi_write_png(output_image4.c_str(), width, height, 1,xImageUC.data(), width) == 0) {
        std::cerr << "Error: Could not save x vector image" << std::endl;
        return 1;
     }

    //STEP 10
    
    SparseMatrix<double> A3 = buildEdgeMatrix(height, width);
    std::cout << "Is A3 symmetric? " << (A3.isApprox(A3.transpose()) ? "Yes" : "No") << std::endl;

    // STEP 11

    VectorXd g_edge = A3 * v;
    Matrix<double, Dynamic, Dynamic, RowMajor> edgeImage = Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(g_edge.data(), height, width);

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> edgeImageUC(height, width);
    edgeImageUC = edgeImage.unaryExpr([](double val) -> unsigned char {
    double c = std::min(std::max(val, 0.0), 255.0);
    return static_cast<unsigned char>(c);
    });
    stbi_write_png("Immagine_con_A3.png", width, height, 1, edgeImageUC.data(), width);


    // STEP 12
SparseMatrix<double> I(height*width,height*width);
    I.setIdentity();
    SparseMatrix<double> Z = (3*I)+A3;
    SparseMatrix<double> checkZ = SparseMatrix<double>(Z.transpose()) - Z;
    std::cout << "Is (3I+A3) symmetric? " << ((checkZ.norm()==0) ? "Yes" : "No") << std::endl;
    
    Eigen::DiagonalPreconditioner<double> D(Z); // Create diag preconditioner

    Eigen::BiCGSTAB<SparseMatrix<double>> BiCG;
    BiCG.compute(Z);
    BiCG.setTolerance(1e-8);
    Eigen::VectorXd y = BiCG.solve(w);

    std::cout << "Eigen native BiCG" << std::endl;
    std::cout << "#iterations: " << BiCG.iterations() << std::endl;
    std::cout << "relative residual: " << BiCG.error() << std::endl;
    
    // STEP 13

    int m = A3.rows();
    Eigen::saveMarketVector(y, "vectory.mtx");
    FILE * out2 =fopen("vectory.mtx","w");
    fprintf(out2,"%%%%MatrixMarket vector coordinate real general\n");
    fprintf(out2,"%d\n", m);
    for (int i=0; i<m; i++) {
        fprintf(out2,"%d %f\n", i ,y(i));
    }
    fclose(out2);

    // Salvataggio immagine vettore x

    Matrix<double, Dynamic, Dynamic, RowMajor> yImage =  Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(y.data(), height, width);
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> yImageUC(height, width);
    yImageUC = yImage.unaryExpr([](double val) -> unsigned char {
            double c = std::min(std::max(val, 0.0), 255.0);
            return static_cast<unsigned char>(c);
            });
    const std::string output_image6 = "Immagine_con_Y.png";
    if (stbi_write_png(output_image6.c_str(), width, height, 1,yImageUC.data(), width) == 0) {
        std::cerr << "Error: Could not save y vector image" << std::endl;
        return 1;
     }

return 0;

}
