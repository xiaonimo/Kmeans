#ifndef KDTREE_H
#define KDTREE_H
#include <cstdlib>
#include <vector>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <iostream>

typedef std::size_t index_t;
typedef std::vector<index_t> indexes_t;

template<typename data_t>
class _cmp {
public:
    bool operator()(const std::pair<data_t, index_t>& p1, const std::pair<data_t, index_t>& p2) {
        return p1.first < p2.first;
    }
};

template<typename data_t>
struct TreeNode {
    std::size_t dim;
    data_t val;
    bool visited;
    std::vector<std::size_t> index_set;  //当前分支中的数据集，只存储index
    struct TreeNode* right;
    struct TreeNode* left;
    struct TreeNode* up;
    inline TreeNode(std::size_t _dim, data_t _val):dim(_dim), val(_val), visited(false), index_set(std::vector<std::size_t>()),
        right(nullptr), left(nullptr), up(nullptr){}
};

template<typename data_t>
class KDTree {
public:
    typedef std::vector<data_t> point_t;
    typedef std::vector<point_t> points_t;

public:
    inline KDTree(points_t _dataset):dataset(_dataset){}
    void build_KDTree(const unsigned int max_leaf_size=1);
    indexes_t search(const unsigned int topk=1);
    void print();

private:
    index_t _get_max_variance_dim(const indexes_t&);
    std::pair<data_t, index_t> _get_split_val_index(const index_t dim, const indexes_t& indexes);
    TreeNode<data_t>* _build_KDTree(const indexes_t&, const unsigned int max_leaf_size);
    void _print(TreeNode<data_t>*);

private:
    TreeNode<data_t>* head;
    points_t dataset;
};

template<typename data_t>
void KDTree<data_t>::build_KDTree(const unsigned int max_leaf_size) {
    indexes_t indexes;
    for (index_t i=0; i<dataset.size(); ++i) indexes.push_back(i);
    head = _build_KDTree(indexes, max_leaf_size);
}

template<typename data_t>
TreeNode<data_t>* KDTree<data_t>::_build_KDTree(const indexes_t& indexes, const unsigned int max_leaf_size) {
    //处理空数据集
    if (indexes.empty()) {
        return nullptr;
    }
    //处理叶节点
    if (indexes.size() <= max_leaf_size) {
        TreeNode<data_t> *_head = new TreeNode<data_t>(data_t(-1), data_t(-1));
        for (auto __index:indexes) _head->index_set.push_back(__index);
        return _head;
    }
    //创建头结点
    index_t _dim = _get_max_variance_dim(indexes);
    std::pair<data_t, index_t> _val_index = _get_split_val_index(_dim, indexes);
    TreeNode<data_t>* _head = new TreeNode<data_t>(_dim, _val_index.first);
    _head->index_set.push_back(_val_index.second);

    //递归创建左子树、右子树
    indexes_t _left_indexes, _right_indexes;
    for (index_t i=0; i<indexes.size(); ++i) {
        index_t __index = indexes[i];
        if (dataset[__index][_dim]<_val_index.first) _left_indexes.push_back(__index);
        else _right_indexes.push_back(__index);
    }
    _head->left = _build_KDTree(_left_indexes, max_leaf_size);
    _head->left->up = _head;
    _head->right = _build_KDTree(_right_indexes, max_leaf_size);
    _head->right->up = _head;

    //递归完成后，返回头结点
    return _head;
}

template<typename data_t>
index_t KDTree<data_t>::_get_max_variance_dim(const indexes_t& indexes) {
    if (indexes.empty()) return index_t(-1);
    std::vector<double> _every_dim_mean(dataset[0].size(), data_t(0));
    std::vector<double> _every_dim_variance(dataset[0].size(), data_t(0));
    for (auto _index:indexes) {
        for (index_t _dim=0; _dim<dataset[0].size(); ++_dim) {
            _every_dim_mean[_dim] += dataset[_index][_dim]/indexes.size();
        }
    }
    for (auto _index:indexes) {
        for (index_t _dim=0; _dim<dataset[0].size(); ++_dim) {
            _every_dim_variance[_dim] += std::pow(dataset[_index][_dim]-_every_dim_mean[_dim], 2);
        }
    }
    index_t res = -1;
    double _variance = std::numeric_limits<double>::min();
    for (index_t _dim=0; _dim<dataset[0].size(); ++_dim) {
        if (_every_dim_variance[_dim]>_variance) {
            _variance = _every_dim_variance[_dim];
            res = _dim;
        }
    }
    return res;
}

template<typename data_t>
std::pair<data_t, index_t> KDTree<data_t>::_get_split_val_index(const index_t dim, const indexes_t& indexes) {
    if (dim<0) throw std::out_of_range("parameter dim is illeagal!");
    std::vector<std::pair<data_t, index_t>> _vec;
    for (auto _index:indexes) _vec.push_back(std::make_pair(dataset[_index][dim], _index));
    sort(std::begin(_vec), std::end(_vec), _cmp<data_t>());
    return _vec[_vec.size()/2];
}

template<typename data_t>
void KDTree<data_t>::print() {
    _print(head);
}

template<typename data_t>
void KDTree<data_t>::_print(TreeNode<data_t> *head) {
    if (!head) return;
    std::cout << head->val <<std::endl;
    _print(head->left);
    _print(head->right);
}

template<typename data_t>
indexes_t KDTree<data_t>::search(const unsigned int topk) {
    if (topk<=0) return indexes_t();

}
#endif // KDTREE_H
