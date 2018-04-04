#ifndef KMEANS_HPP
#define KMEANS_HPP
#include <vector>
#include <string>
#include <random> // for _init_centers()
#include <stdexcept>
#include <cstdio>
#include <limits>
#include <iostream>
#include <unordered_map>

namespace wzt {

typedef std::size_t index_t;
typedef std::vector<index_t> indexes_t;

template<typename data_t>
class Kmeans {
public:
    typedef std::vector<data_t> point_t;
    typedef std::vector<point_t> points_t;

public:
    //inline Kmeans(){}
    inline Kmeans(const points_t _dataset):data_set(_dataset), data_dim(data_set[0].size()), data_cnt(data_set.size()){}
    //用户可以传入自定义的聚类中心
    void cluster(unsigned int, points_t _centers=points_t(), std::string _Metric="L2", bool _verbose=true);

public:
    points_t data_set;
    unsigned int data_dim;
    unsigned int data_cnt;

    unsigned int n_clusters;
    std::vector<std::vector<double>> centers;
    bool verbose;
    std::string Metric;

private:
    void _init_params(unsigned int, std::string, bool);
    void _init_centers(points_t);
    void _cluster();
    index_t _get_nearest_center(const point_t&);
    void _update_centers();
    double _metric(point_t const&, point_t const&);

public:
    std::vector<std::vector<std::size_t>> cluster_res;
    double _sum_dist_cur=0.;
    double _sum_dist_last=0.;
};

template<typename data_t>
void Kmeans<data_t>::cluster(unsigned int _n_clusters, Kmeans<data_t>::points_t _centers, std::string _Metric, bool _verbose) {
    _init_params(_n_clusters, _Metric, _verbose);
    _init_centers(_centers);

    index_t _itr = 20;
    while (_itr-->0) {
        _cluster();
        _update_centers();

        if (verbose) {
            std::cout << _itr << "\t" << _sum_dist_cur-_sum_dist_last<< std::endl;
            for (auto v:cluster_res) std::cout << v.size() <<"\t";
            std::cout << std::endl;
        }
    }
}

template<typename data_t>
void Kmeans<data_t>::_init_params(unsigned int _n_clusters, std::string _Metric, bool _verbose) {
    n_clusters = _n_clusters;
    Metric = _Metric;
    verbose = _verbose;

}

template<typename data_t>
void Kmeans<data_t>::_init_centers(typename Kmeans<data_t>::points_t _centers) {
    if (_centers.empty()) {
        std::mt19937 _gen;
        std::uniform_int_distribution<int> _random_index(0, data_cnt-1);
        centers = std::vector<std::vector<double>>(n_clusters);

        std::unordered_map<int, bool> hmap;
        for (index_t i=0; i<n_clusters;) {
            index_t __index = _random_index(_gen);
            if (hmap[__index]) continue;
            hmap[__index] = true;
            centers[i] = data_set[__index];
            ++i;
        }
    } else {
        if (n_clusters != _centers.size())
            throw std::invalid_argument("n_clusters != _centers.size()");
        centers = _centers;
    }
}

template<typename data_t>
void Kmeans<data_t>:: _cluster() {
    _sum_dist_last = _sum_dist_cur;
    cluster_res = std::vector<std::vector<index_t>>(n_clusters, std::vector<index_t>());
    for (index_t _index=0; _index<data_cnt; ++_index) {
        index_t _center_index = _get_nearest_center(data_set[_index]);
        cluster_res[_center_index].push_back(_index);
        _sum_dist_cur += _metric(centers[_center_index], data_set[_index]);
    }
}

template<typename data_t>
index_t Kmeans<data_t>::_get_nearest_center(const point_t &p) {
    index_t _res = -1;
    double _dist = std::numeric_limits<double>::max();
    for (index_t _index=0; _index<n_clusters; ++_index) {
        double __dist = _metric(p, centers[_index]);
        if (__dist >= _dist) continue;
        _dist = __dist;
        _res = _index;
    }
    return _res;
}

template<typename data_t>
void Kmeans<data_t>::_update_centers() {
    //每个聚类中心
    for (index_t c=0; c<n_clusters; ++c) {
        point_t _new_center(data_dim, data_t(0));
        index_t _cluster_point_num = cluster_res[c].size();
        //每个簇中的点
        for (index_t p=0; p<_cluster_point_num; ++p) {
            //每个维度
            for (index_t _dim=0; _dim<data_dim; ++_dim) _new_center[_dim] += data_set[cluster_res[c][p]][_dim]/double(_cluster_point_num);
        }
        centers[c] = _new_center;
    }
}

template<typename data_t>
double Kmeans<data_t>::_metric(typename Kmeans<data_t>::point_t const &p1, typename Kmeans<data_t>::point_t const &p2) {
    double _dist = 0;
    index_t _data_dim = p1.size();
    if (Metric == std::string("L2")) {
        for (index_t _dim=0; _dim<_data_dim; ++_dim) _dist += std::pow(p1[_dim]-p2[_dim], 2);
    }
    return _dist;
}


template<typename data_t>
void read_train_mnist(std::vector<std::vector<data_t>>& X, std::vector<std::vector<data_t>>& Y, std::string filename) {
    if (X.size() != Y.size()) throw std::invalid_argument(" X and Y'size shoule be same!");
    int num = X.size();

    std::freopen(filename.c_str(), "r", stdin);
    data_t val = 0.;
    for (int i=0; i<num; ++i) {
        for (int j=0; j<784+1; ++j) {
            scanf("%d,", &val);
            if (j == 0) Y[i][0] = val;
            else X[i][j-1] = val;
        }
    }
    std::fclose(stdin);
    std::freopen("CON", "r", stdin);
    std::cout << "read mnist data finished" << std::endl;
}

template<typename data_t>
void data_normalization(std::vector<std::vector<data_t>>& X) {
    for (index_t r=0; r<X.size(); ++r) {
        for (index_t c=0; c<X[0].size(); ++c)  {
            X[r][c]=X[r][c]?1:0;
        }
    }
}

template<typename data_t>
typename std::vector<std::vector<data_t>> data_pooling(const std::vector<std::vector<data_t>>& X) {
    index_t r = X.size();
    index_t c = X[0].size();
    index_t sz = int(sqrt(c));

    auto res = std::vector<std::vector<data_t>>(r);
    for (index_t i=0; i<r; ++i) {
        for (index_t m=0; m<sz-1; m+=2) {
            for (index_t n=0; n<sz-1; n+=2) {
                data_t _res = X[i][m*sz+n]+X[i][m*sz+n+1]+X[i][m*sz+n+sz]+X[i][m*sz+n+sz+1]>0?1:0;
                res[i].push_back(_res);
            }
        }
    }
    return res;
}

}
#endif // KMEANS_HPP
