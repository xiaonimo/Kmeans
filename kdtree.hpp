#ifndef KDTREE_HPP
#define KDTREE_HPP
#include <cstdlib>
#include <vector>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace wzt {

typedef std::size_t index_t;
typedef std::vector<index_t> indexes_t;

template<typename data_t>
class _cmp {
public:
    bool operator()(const std::pair<index_t, data_t>& p1, const std::pair<index_t, data_t>& p2) {
        return p1.second < p2.second;
    }
};

template<typename data_t>
struct TreeNode {
    std::size_t dim;
    data_t val;
    indexes_t index_set;  //当前分支中的数据集，只存储index
    struct TreeNode* right;
    struct TreeNode* left;
    inline TreeNode(std::size_t _dim, data_t _val):dim(_dim), val(_val), index_set(indexes_t()),
        right(nullptr), left(nullptr){}
};

template<typename data_t>
class KDTree {
public:
    typedef std::vector<data_t> point_t;
    typedef std::vector<point_t> points_t;

public:
    inline KDTree(points_t _dataset):dataset(_dataset), data_dim(_dataset[0].size()), data_cnt(_dataset.size()){}
    void build_KDTree(const unsigned int max_leaf_size=1);
    points_t search(const point_t&, const unsigned int topk=1);
    indexes_t search_index(const point_t&, const unsigned int topk=1);
    void print();
    unsigned tree_min_depth();
    unsigned tree_max_depth();

private:
    index_t _get_max_variance_dim(const indexes_t&);
    std::pair<index_t, data_t> _get_split_val_index(const index_t dim, const indexes_t& indexes);
    //TreeNode<data_t>* _build_KDTree(const indexes_t&, const unsigned int max_leaf_size);
    void _build_KDTree(TreeNode<data_t>* &, const indexes_t&, const unsigned int max_leaf_size);
    void _search(const point_t&, const unsigned int, TreeNode<data_t>*, std::vector<std::pair<index_t, data_t>> &res);
    void _print(const TreeNode<data_t>*);
    void _print(const point_t p);
    data_t Metric(point_t, point_t);
    unsigned _tree_max_depth(TreeNode<data_t>*);
    unsigned _tree_min_depth(TreeNode<data_t>*);

private:
    TreeNode<data_t>* head;
    points_t dataset;
    const index_t data_dim;
    const index_t data_cnt;
};

template<typename data_t>
void KDTree<data_t>::build_KDTree(const unsigned int max_leaf_size) {
    indexes_t indexes;
    for (index_t i=0; i<dataset.size(); ++i) indexes.push_back(i);
    _build_KDTree(head, indexes, max_leaf_size);
    std::cout << "KDTree build finished!" <<std::endl;
}

/*
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
    std::pair<index_t, data_t> _val_index = _get_split_val_index(_dim, indexes);
    TreeNode<data_t>* _head = new TreeNode<data_t>(_dim, _val_index.second);
    _head->index_set.push_back(_val_index.first);

    //递归创建左子树、右子树
    indexes_t _left_indexes, _right_indexes;
    for (index_t i=0; i<indexes.size(); ++i) {
        index_t __index = indexes[i];
        //处理val与head->val相同的情况
        if (__index == _head->index_set.front()) continue;
        if (dataset[__index][_dim]<=_head->val) _left_indexes.push_back(__index);
        else _right_indexes.push_back(__index);
    }
    _head->left = _build_KDTree(_left_indexes, max_leaf_size);
    _head->right = _build_KDTree(_right_indexes, max_leaf_size);

    //递归完成后，返回头结点
    return _head;
}*/

template<typename data_t>
void KDTree<data_t>::_build_KDTree(TreeNode<data_t>* &_head, const indexes_t& indexes, const unsigned int max_leaf_size) {
    //处理空数据集
    if (indexes.empty()) {
        return;
    }
    //处理叶节点
    if (indexes.size() <= max_leaf_size) {
        _head = new TreeNode<data_t>(data_t(-1), data_t(-1));
        for (auto __index:indexes) _head->index_set.push_back(__index);
        return;
    }
    //创建头结点
    index_t _dim = _get_max_variance_dim(indexes);
    std::pair<index_t, data_t> _val_index = _get_split_val_index(_dim, indexes);
    _head = new TreeNode<data_t>(_dim, _val_index.second);
    _head->index_set.push_back(_val_index.first);

    //递归创建左子树、右子树
    indexes_t _left_indexes, _right_indexes;
    for (index_t i=0; i<indexes.size(); ++i) {
        index_t __index = indexes[i];
        //处理val与head->val相同的情况
        if (__index == _head->index_set.front()) continue;
        if (dataset[__index][_dim]<=_head->val) _left_indexes.push_back(__index);
        else _right_indexes.push_back(__index);
    }
    _build_KDTree(_head->left, _left_indexes, max_leaf_size);
    _build_KDTree(_head->right, _right_indexes, max_leaf_size);

    //递归完成后，返回
    return;
}

template<typename data_t>
index_t KDTree<data_t>::_get_max_variance_dim(const indexes_t& indexes) {
    if (indexes.empty()) return index_t(-1);
    std::vector<double> _every_dim_mean(data_dim, data_t(0));
    std::vector<double> _every_dim_variance(data_dim, data_t(0));
    for (auto _index:indexes) {
        for (index_t _dim=0; _dim<data_dim; ++_dim) {
            _every_dim_mean[_dim] += dataset[_index][_dim]/indexes.size();
        }
    }
    for (auto _index:indexes) {
        for (index_t _dim=0; _dim<data_dim; ++_dim) {
            _every_dim_variance[_dim] += std::pow(dataset[_index][_dim]-_every_dim_mean[_dim], 2);
        }
    }
    index_t res = -1;
    double _variance = std::numeric_limits<double>::min();
    for (index_t _dim=0; _dim<data_dim; ++_dim) {
        if (_every_dim_variance[_dim]>_variance) {
            _variance = _every_dim_variance[_dim];
            res = _dim;
        }
    }
    return res;
}

template<typename data_t>
std::pair<index_t, data_t> KDTree<data_t>::_get_split_val_index(const index_t dim, const indexes_t& indexes) {
    if (dim<0) throw std::out_of_range("parameter dim is illeagal!");
    std::vector<std::pair<index_t, data_t>> _vec;
    for (auto _index:indexes) _vec.push_back(std::make_pair(_index, dataset[_index][dim]));
    std::sort(std::begin(_vec), std::end(_vec), _cmp<data_t>());
    return _vec[_vec.size()/2];
}

template<typename data_t>
void KDTree<data_t>::print() {
    _print(head);
}

template<typename data_t>
void KDTree<data_t>::_print(const TreeNode<data_t> *_head) {
    if (!_head) return;
    _print(_head->left);
    std::cout << _head->val << "\t";
    _print(_head->right);
}

template<typename data_t>
void KDTree<data_t>::_print(const point_t p) {
    std::cout << "[" ;
    for (auto e:p) std::cout << e <<",";
    std::cout << "]";
}

template<typename data_t>
data_t KDTree<data_t>::Metric(typename KDTree<data_t>::point_t p1, typename KDTree<data_t>::point_t p2) {
    data_t _dist = data_t(0);
    //L2 Metric
    for (index_t _dim=0; _dim<p1.size(); _dim++) _dist += std::pow(p1[_dim]-p2[_dim], 2);
    return _dist;
}

template<typename data_t>
typename KDTree<data_t>::points_t KDTree<data_t>::search(const point_t &p, const unsigned int topk) {
    if (topk<=0 || topk>data_cnt) throw std::invalid_argument("Parameter topk is invalid!\n");

    std::vector<std::pair<index_t, data_t>> _indexes_dists;
     _search(p, topk, head, _indexes_dists);
    points_t res;
    for (auto _index_dist:_indexes_dists) {
        res.push_back(dataset[_index_dist.first]);
        _print(dataset[_index_dist.first]);
        std::cout << "\t" << _index_dist.second<<std::endl;
    }
    return res;
}

template<typename data_t>
indexes_t KDTree<data_t>::search_index(const point_t&  p, const unsigned int topk) {
    if (topk<=0 || topk>data_cnt) throw std::invalid_argument("Parameter topk is invalid!\n");

    std::vector<std::pair<index_t, data_t>> _indexes_dists;
     _search(p, topk, head, _indexes_dists);
     indexes_t res;
     for (auto _index_dist:_indexes_dists) {
         res.push_back(_index_dist.first);
     }
     return res;
}

template<typename data_t>
void KDTree<data_t>::_search(const point_t &p, const unsigned int topk, TreeNode<data_t> *head,
                             std::vector<std::pair<index_t, data_t>> &res) {
    if (!head) return;
    //std::cout << "res_size" << res.size() << std::endl;

    //std::vector<std::pair<indexes_t, data_t>> cur_res;
    if (head->dim != std::size_t(-1)) {
    //注意这里的分支原则，要与建树时一致
    if (p[head->dim] <= head->val) {
        _search(p, topk, head->left, res);
        if (res.size()<topk || res.empty() || std::abs(dataset[res.back().first][head->dim]-head->val)<res.back().second) {
            _search(p, topk, head->right, res);
        }
    } else {
        _search(p, topk, head->right, res);
        if (res.size()<topk || res.empty() || std::abs(dataset[res.back().first][head->dim]-head->val)<res.back().second) {
            _search(p, topk, head->left, res);
        }
    }
    }// end if(head->dim != -1)

    //先把res灌满
    index_t i = index_t(0);
    while (res.size()<topk && i<head->index_set.size()) {
        data_t _dist = Metric(p, dataset[head->index_set[i]]);
        res.push_back(std::make_pair(head->index_set[i], _dist));
        i++;
    }
    //排序
    sort(std::begin(res), std::end(res), _cmp<data_t>());
    //剩余的点，按照插入排序的方式进入res
    while (i<head->index_set.size()) {
        //先处理res中最后一个点
        index_t j = index_t(topk-1);
        data_t __dist = Metric(p, dataset[head->index_set[i]]);
        if (__dist < res[j].second) {
            if (j==0) res[j]=std::make_pair(head->index_set[i], __dist);
            else res[j] = res[j-1];
        }
        j--;
        //循环处理之前的点
        while (j>=0 && __dist<res[j].second) {
            res[j+1] = res[j];
            j--;
        }
        res[j+1] = std::make_pair(head->index_set[i], __dist);
        i++;
    }
    return;
}

template<typename data_t>
unsigned KDTree<data_t>::tree_max_depth() {
    return _tree_max_depth(head);
}

template<typename data_t>
unsigned KDTree<data_t>::_tree_max_depth(TreeNode<data_t>* head) {
    return head?std::max(_tree_max_depth(head->left), _tree_max_depth(head->right))+1:0;
}

template<typename data_t>
unsigned KDTree<data_t>::tree_min_depth() {
    return _tree_min_depth(head);
}

template<typename data_t>
unsigned KDTree<data_t>::_tree_min_depth(TreeNode<data_t>* head) {
    if (!head) return 0;
    if (head->left && !head->right) return _tree_min_depth(head->left);
    if (!head->left && head->right) return _tree_min_depth(head->right);
    return std::min(_tree_min_depth(head->left), _tree_min_depth(head->right))+1;
}

}// end namespace


#endif // KDTREE_HPP
