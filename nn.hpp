#ifndef NN_HPP
#define NN_HPP

#include <random>
#include <vector>
#include <string>
#include <stdexcept>

#include "func.hpp"


typedef unsigned c_uint;

class Mat {
public:
    Mat(){}
    Mat(const unsigned rows, const unsigned cols):rows(rows), cols(cols) {
        data = std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0));
    }
    std::vector<double>& operator [](const unsigned row_index) {
        return data.at(row_index);
    }
    Mat& operator =(const Mat& b) {
        if (this == &b) return *this;
        this->rows = b.rows;
        this->cols = b.cols;
        this->data = b.data;
        return *this;
    }

public:
    c_uint rows, cols;
    std::vector<std::vector<double>> data;
};

class Layer {
public:
    Layer(){}
    Layer(const std::string lname, const std::string aname): NAME(lname), ACTIVE(aname) {}
    Layer& operator =(const Layer &b) {
        if (this == &b) return *this;
        this->NAME = b.NAME;
        this->ACTIVE = b.ACTIVE;
        return *this;
    }

public:
    std::string NAME;     //conv, mpool, apool, full, input, output;
    std::string ACTIVE;   //sigmoid, relu, tanh;
};

class Conv_Layer: public Layer {
public:
    Conv_Layer(const std::string lname, const std::string aname, c_uint f_cnt, c_uint f_rows, c_uint f_cols, c_uint k_rows, c_uint k_cols, c_uint stride):
    Layer(lname, aname), F_CNT(f_cnt), K_CNT(f_cnt), STRIDE(stride), F_ROWS(f_rows), F_COLS(f_cols), K_ROWS(k_rows), K_COLS(k_cols){
        data = std::vector<Mat>(F_CNT, Mat(F_ROWS, F_COLS));
        error = std::vector<Mat>(F_CNT, Mat(F_ROWS, F_COLS));
        kernels = std::vector<Mat>(K_CNT, Mat(K_ROWS, K_COLS));
        d_kernels = std::vector<Mat>(K_CNT, Mat(K_ROWS, K_COLS));
        bias = std::vector<double>(F_CNT, 0);
        d_bias = std::vector<double>(F_CNT, 0);
    }

public:
    c_uint F_CNT, K_CNT, STRIDE;

    c_uint F_ROWS, F_COLS;
    std::vector<Mat> data, error;

    c_uint K_ROWS, K_COLS;
    std::vector<Mat> kernels, d_kernels;

    std::vector<double> bias, d_bias;
};

class Full_Layer: public Layer {
public:
    Full_Layer(){}
    Full_Layer(c_uint prev_layer_neural_cnt, c_uint cur_layer_neural_cnt,
               const std::string lname=std::string("full_layer"), const std::string aname=std::string("sigmoid")):
        Layer(lname, aname), N_CNT(cur_layer_neural_cnt){
        data  = std::vector<double>(N_CNT, 0);
        error = std::vector<double>(N_CNT, 0);

        W  = Mat(prev_layer_neural_cnt, cur_layer_neural_cnt);
        dW = Mat(prev_layer_neural_cnt, cur_layer_neural_cnt);

        B  = std::vector<double>(N_CNT, 0);
        dB = std::vector<double>(N_CNT, 0);
    }
    Full_Layer& operator =(const Full_Layer& b) {
        if (this == &b) return *this;
        Layer::operator =(b);
        this->N_CNT = b.N_CNT;
        this->data = b.data;
        this->error = b.error;
        this->W = b.W;
        this->dW = b.dW;
        this->B = b.B;
        this->dB = b.dB;
        return *this;
    }

public:
    c_uint N_CNT;

    std::vector<double> data, error;

    Mat W, dW;

    std::vector<double> B, dB;
};

class Pool_Layer: public Layer {
public:
    Pool_Layer(const std::string lname, const std::string aname, c_uint f_cnt, c_uint f_rows, c_uint f_cols, c_uint psz, c_uint stride):
    Layer(lname, aname), F_CNT(f_cnt), P_SZ(psz), STRIDE(stride), F_ROWS(f_rows), F_COLS(f_cols){
        data = std::vector<Mat>(F_CNT, Mat(F_ROWS, F_COLS));
        error = std::vector<Mat>(F_CNT, Mat(F_ROWS, F_COLS));
    }

public:
    c_uint F_CNT, P_SZ, STRIDE;

    c_uint F_ROWS, F_COLS;
    std::vector<Mat> data, error;
};

class Active {
public:
    static double active(double x) {
        //return std::tanh(x);
        return 1/(1+std::exp(-x));
    }
    static double d_active(double y) {
        //return 1 - y*y;
        return y*(1-y);
    }
};


class NN {
private:
    typedef std::initializer_list<Layer> list;
    typedef std::initializer_list<Full_Layer> Flist;
    typedef std::vector<std::vector<double>> _mat;
public:
    NN(Flist layers, _mat&train_x, _mat&train_y, _mat&test_x, _mat&test_y):
        Layers(layers), train_x(train_x), train_y(train_y), test_x(test_x), test_y(test_y) {
        //X = Full_Layer(0, 784);
        Y = Full_Layer(0, 10);
        init_weights();
    }
    void fit();
    //TODO
    //unsigned predict(const Full_Layer&);
    //std::vector<unsigned> predict()

    //Tmp
    void predict() {
        unsigned ca = 0;
        for (unsigned i=0; i<test_x.size(); ++i) {
            setX(test_x[i]);
            forword_flow();
            ca += ff::argmax(Y.data)==ff::argmax(Layers.back().data);
        }
        std::cout << " accuracy:" << double(ca)/test_x.size() <<std::endl;
    }

public:
    std::vector<Full_Layer> Layers;
    _mat &train_x, &train_y, &test_x, &test_y;

private:
    void init_weights();
    void forword_flow();
    void backword_flow();
    void update_weights();

    void forword_flow_full(const Full_Layer&, Full_Layer&);
    void forword_flow_conv(Layer&, Layer&);
    void forword_flow_pool(Layer&, Layer&);

    void backword_flow_full(Full_Layer&, Full_Layer&, bool isOutput);
    void backword_flow_conv(Layer&, Layer&);
    void backword_flow_pool(Layer&, Layer&);

    void conv(const Mat&, const Mat&, Mat&, double b);
    void kronecker(const Mat&, const Mat&, Mat&);

    void setX(const std::vector<double> tx) {
        Layers[0].data = tx;
    }
    void setY(const std::vector<double> ty) {
        Y.data = ty;
    }

private:
    Full_Layer Y;
    const double learnint_rate = 0.001;
};

void
NN::forword_flow() {
    for (unsigned i=0; i<Layers.size()-1; ++i) {
        forword_flow_full(Layers[i], Layers[i+1]);
    }
}

void
NN::backword_flow() {
    for (unsigned i=1; i<Layers.size(); ++i) {
        if (i == 1) {
            backword_flow_full(Layers.back(), Y, true);
        } else {
            //std::cout << Layers.size()-i-1 << " "  << Layers.size()-i <<std::endl;
            backword_flow_full(Layers[Layers.size()-i-1], Layers[Layers.size()-i], false);
        }
    }
}

void
NN::init_weights() {
    std::mt19937 gen;
    std::normal_distribution<double> normal(-0.001, 0.001);
    for (Full_Layer &_layer:Layers) {
        for (unsigned i=0; i<_layer.W.rows; ++i) {
            for (unsigned j=0; j<_layer.W.cols; ++j) {
                _layer.W[i][j] = normal(gen);
            }
        }
    }
    std::cout << "init weights finished!" << std::endl;
}

void
NN::update_weights() {
    for (Full_Layer &_layer:Layers) {
        for (unsigned r=0; r<_layer.W.rows; ++r) {
            for (unsigned c=0; c<_layer.W.cols; ++c) {
                _layer.W[r][c] -= learnint_rate * _layer.dW[r][c];
            }
        }
        for (unsigned i=0; i<_layer.N_CNT; ++i) {
            _layer.B[i] -= learnint_rate * _layer.dB[i];
        }
    }
}

void
NN::fit() {
    for (unsigned e=0; e<20; ++e) {
        for (unsigned i=0; i<train_x.size(); ++i) {
            setX(train_x[i]);
            setY(train_y[i]);
            for (unsigned j=0; j<100; ++j) {
                forword_flow();
                backword_flow();
                update_weights();
            }
        }
        std::cout << "epoch:" << e;
        predict();
    }
}

void
NN::forword_flow_full(const Full_Layer &prev_layer, Full_Layer &cur_layer) {
    for (unsigned i=0; i<cur_layer.N_CNT; ++i) {
        double sum = 0.;
        for (unsigned j=0; j<prev_layer.N_CNT; ++j) {
            sum += prev_layer.data[j] * cur_layer.W[j][i];
        }
        cur_layer.data[i] = Active::active(sum + cur_layer.B[i]);
    }
}

//void
//NN::forword_flow_conv(Layer &prev_layer, Layer &cur_layer) {
//    if (prev_layer.NAME == std::string("input")) {
//        for (unsigned k=0; k<cur_layer.F_CNT; ++k) {
//            conv(prev_layer.data[0], cur_layer.kernels[k], cur_layer.data[k], cur_layer.bias[k]);
//        }
//    } else {
//        if (prev_layer.F_CNT != cur_layer.F_CNT) {
//            throw std::invalid_argument("The number of prev_layer's F_CNT is not equal to the cur_layer's");
//        }
//        for (unsigned k=0; k<cur_layer.F_CNT; ++k) {
//            conv(prev_layer.data[k], cur_layer.kernels[k], cur_layer.data[k], cur_layer.bias[k]);
//        }
//    }
//}

void
NN::backword_flow_full(Full_Layer &prev_layer, Full_Layer &cur_layer, bool isOutput) {
    if (isOutput) {
        if (prev_layer.N_CNT != cur_layer.N_CNT) {
            throw std::invalid_argument("The number of output_layer's N_CNT is not equal to prev_layer's!");
        }
        double loss = 0;
        for (unsigned i=0; i<prev_layer.N_CNT; ++i) {
            prev_layer.error[i] = (prev_layer.data[i] - cur_layer.data[i]) * Active::active(prev_layer.data[i]);
            loss += std::pow(prev_layer.data[i] - cur_layer.data[i], 2);
        }
        std::cout << "loss:" << loss <<std::endl;
    } else {
        //update prev_layer's error
        if (prev_layer.W.rows != 0) {
            for (unsigned i=0; i<prev_layer.N_CNT; ++i) {
                double sum = 0.;
                for (unsigned j=0; j<cur_layer.N_CNT; ++j) {
                    sum += cur_layer.W[i][j] * cur_layer.error[j];
                }
                prev_layer.error[i] = sum * Active::d_active(prev_layer.data[i]);
            }
        }
        //update cur_layer's dw, db
        for (unsigned i=0; i<prev_layer.N_CNT; ++i) {
            for (unsigned j=0; j<cur_layer.N_CNT; ++j) {
                cur_layer.dW[i][j] = prev_layer.data[i] * cur_layer.error[j];
            }
        }
        for (unsigned i=0; i<cur_layer.N_CNT; ++i) {
            cur_layer.dB[i] = cur_layer.error[i];
        }
    }
}







































#endif // NN_HPP
