#include "kdtree.hpp"
#include "kmeans.hpp"
#include "knn.hpp"

int main() {

    /*
    std::vector<std::vector<int>> X(10000, std::vector<int>(784, 0));
    std::vector<std::vector<int>> Y(10000, std::vector<int>(1, 0));
    wzt::read_train_mnist<int>(X, Y, "train.csv");

    auto train_x = std::vector<std::vector<int>>(std::begin(X), std::begin(X)+8000);
    auto train_y = std::vector<std::vector<int>>(std::begin(Y), std::begin(Y)+8000);
    auto test_x = std::vector<std::vector<int>>(std::begin(X)+8000, std::end(X));
    auto test_y = std::vector<std::vector<int>>(std::begin(Y)+8000, std::end(Y));

    wzt::Knn<int> k;
    k.set_train_data(train_x, train_y);
    k.set_test_data(test_x, test_y);
    k.predict();
    */

    wzt::KDTree<int> a(std::vector<std::vector<int>>{{1,1},{2,2},{3,3},{3,4},{4,3},{4,4},{4,5},{9,9},{2,6}});
    a.build_KDTree();
    a.print();
    a.search(std::vector<int>{3,3}, 3);

    /*

    wzt::data_normalization<double>(X);
    auto _X = wzt::data_pooling(wzt::data_pooling(X));
    wzt::Kmeans<double> k(_X);
    //std::cout << "funk";
    k.cluster(10);
    for (wzt::index_t i=0; i<k.n_clusters; ++i) {
        std::cout << "<" << i << ">"<<std::endl;
        for (auto index:k.cluster_res[i]) {
            std::cout<< Y[index][0] <<",";
        }
        std::cout <<std::endl<<std::endl;
    }
    */
}
