#ifndef KNN_HPP
#define KNN_HPP
#include <vector>
#include "kdtree.hpp"

namespace wzt {

typedef std::size_t index_t;
typedef std::vector<index_t> indexes_t;

template<typename data_t>
class Knn {
public:
    typedef std::vector<data_t> point_t;
    typedef std::vector<point_t> points_t;

public:
    inline Knn(){}
    void set_train_data(points_t, points_t);
    void set_test_data(points_t, points_t);
    indexes_t predict();

private:
    points_t train_X, train_Y;
    points_t test_X, test_Y;
    KDTree<data_t> *kdtree;
};

template<typename data_t>
void Knn<data_t>::set_train_data(points_t X, points_t Y) {
    train_X = X;
    train_Y = Y;
    kdtree = new KDTree<data_t>(X);
    kdtree->build_KDTree();
    std::cout << "max_length:" << kdtree->tree_max_depth() << std::endl;
    std::cout << "min_length:" << kdtree->tree_min_depth() << std::endl;
}

template<typename data_t>
void Knn<data_t>::set_test_data(points_t X, points_t Y) {
    test_X = X;
    test_Y = Y;
}

template<typename data_t>
indexes_t Knn<data_t>::predict() {
    indexes_t res;
    int correct_ans=0;
    for (index_t i=0; i<test_X.size(); ++i) {
        auto _search_res = kdtree->search_index(test_X[i]);
        std::cout << _search_res[0] << "\t";
        auto _pred_y = train_Y[_search_res[0]][0];
        auto _real_y = test_Y[i][0];
        //res.push_back(_search_res[0]);
        std::cout << _pred_y << "/" << _real_y <<std::endl;
        correct_ans += (_pred_y == _real_y);
    }
    std::cout << correct_ans/double(test_X.size()) << std::endl;
    return res;
}
}//end namespace


#endif // KNN_HPP
