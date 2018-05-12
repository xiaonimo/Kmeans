#include "func.hpp"
#include "nn.hpp"

int main() {
    const unsigned num = 4000;
    points_t X(num, point_t(784));
    points_t Y(num, point_t(10));
    ff::read_mnist(X, Y, "train.csv");
    ff::data_normalization1(X);

    auto train_x = points_t(std::begin(X), std::begin(X)+int(num*0.9));
    auto train_y = points_t(std::begin(Y), std::begin(Y)+int(num*0.9));

    auto test_x = points_t(std::begin(X)+int(num*0.9), std::end(X));
    auto test_y = points_t(std::begin(Y)+int(num*0.9), std::end(Y));

    Full_Layer f1(0, 784);
    Full_Layer f2(784, 100);
    Full_Layer f3(100, 10);
    NN mlp({f1, f2, f3}, train_x, train_y, test_x, test_y);
    mlp.fit();

    return 0;
}
