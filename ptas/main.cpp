#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <unordered_map>
#include <iomanip>

using namespace std;

const int d = 2;
const int pow2 = 4;
const double sqrtd = 1;
const int c = 2; // desired error rate as 1 + 1/c
int m; // portals per edge, K * c log n, K = 2?
const int r = 10 * c; // number of crossings; should be O(c); multiplying it by any constant makes the algorithm slowen exponentially (being still polynomial over delusional constant factors)

vector<pair<int, int>> compute_portal_pairs() {
    vector<pair<int, int>> portal_pairs;
    const int num_edges = d * pow2 / 2; 
    
    for (int edge_id = 0; edge_id < num_edges; ++edge_id) {
        for (int portal_id = 0; portal_id < m; ++portal_id) {
            portal_pairs.emplace_back(edge_id, portal_id);
        }
    }
    return portal_pairs;
}

vector<pair<int,int> > portal_pairs;

void recursive_k_fillup(int k, vector<pair<int,int> >& res, vector<vector<pair<int,int> > >& answ){
    if(res.size() == k){
        answ.push_back(res);
        return;
    }
    for(auto& u : portal_pairs){
        res.push_back(u);
        recursive_k_fillup(k, res, answ);
        res.pop_back();
    }
}

vector<vector<pair<int,int> > > precompute_portal_configurations() {
    vector<vector<pair<int,int> > > answ;
    vector<pair<int,int> > res;
    for(int k = 0; k <= r; ++k){
        recursive_k_fillup(k, res, answ);
    }
    return answ;
}

vector<vector<pair<int,int> > > possible_configurations;



template<typename T>
struct Point{
    T a[d];
    Point(){
        for(int i = 0; i < d; ++i){
            a[i] = 0;
        }
    }
    Point(const T* arr) {
        for(int i = 0; i < d; ++i){
            a[i] = arr[i];
        }
    }
    template<typename Y>
    Point(const Point<Y>& other) {
        for(int i = 0; i < d; ++i){
            a[i] = other.a[i];
        }
    }

    bool operator == (const Point& other){
        bool eq = 1;
        for(int i = 0; i < d; ++i){
            eq &= other.a[i] == a[i];
        }
        return eq;
    }

    bool operator != (const Point& other) {
        return !(*this == other);
    }

    Point& operator *= (const long double& x){
        for(int i = 0; i < d; ++i){
            a[i] *= x;
        }
    }
    
    Point& operator += (const Point& other){
        for(int i = 0; i < d; ++i){
            a[i] += other.a[i];
        }
    }
    Point& operator -= (const Point& other){
        for(int i = 0; i < d; ++i){
            a[i] -= other.a[i];
        }
    }
};

template<typename T>
long double dist(const Point<T>& a, const Point<T>& b){
    long double res = 0;
    for(int i = 0; i < d; ++i){
        res+= (a.a[i] - b.a[i]) * (a.a[i] - b.a[i]);
    }
    return sqrtl(res);
}

namespace std {
    template<> struct hash< vector<pair<int,int> > >  {
        size_t operator()(const vector<pair<int,int> >& pc) const {
            size_t seed = 0;
            for (const auto& p : pc) {
                seed ^= hash<int>()(p.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                seed ^= hash<int>()(p.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
}


struct Tree_node{
    int point_index;
    int l[d], r[d];
    vector<Tree_node*> children; // each child may split at every side of the partition, at most pow2 * pow2 children
    unordered_map<vector< pair<int,int> >, double> dp;
    bool belongs_to(const Point<int>& a){
        for(int i = 0; i < d; ++i){
            if(a.a[i] < l[i] || a.a[i] > r[i]){
                return 0;
            }
        }
        return 1;
    }
};

struct Permutation { // order of portals visited 
    // each edge is coded by a number 0 to d-1 (axis) and 0 to 2^(d-1) - 1 (starting point) and mapped to (2 ^ d - 1) * axis_id + starting_point_id
    // each portal is coded by a number 0 to m - 1 and indexed by ascension of coordinates
    Tree_node* base;
    vector<pair<int, int> >* p;

    Permutation(Tree_node* base, vector<pair<int,int> >* p): base(base), p(p){}

    pair<Point<int>, Point<int> > get_edge(int edge_id){
        int axis_id = edge_id / (pow2 / 2);
        int axis_mask = edge_id % (pow2 / 2);
        Point<int> answ;
        int i = -1;
        for(int j = 0; j < d; ++j){
            if (j == axis_id){
                continue;
            }
            i += 1;
            if (axis_mask & (1 << i)){
                answ.a[j] = base->r[j];
            }else{
                answ.a[j] = base->l[j];
            }
        }
        Point<int> answ2 = answ;
        answ.a[axis_id] = base->l[axis_id];
        answ2.a[axis_id] = base->r[axis_id];
        return make_pair(answ, answ2);
    }

    vector<int> get_projection(int edge_id){
        vector<int> res;
        for(int i = 0; i < (*p).size(); ++i){
            auto u = (*p)[i];
            if (u.first == edge_id) {
                res.push_back(u.second);
            }
        }
        return res;
    }

    Point<double> get_portal(int edge_id, double portal_id){
        auto edge = get_edge(edge_id);
        edge.second -= edge.first;
        double coef = portal_id / m;
        edge.second *= coef;
        edge.first += edge.second;
        return edge.first;
    }
};

bool Match(pair<Point<int>, Point<int>> first_edge, const vector<int>& first_proj,
    pair<Point<int>, Point<int>> second_edge, const vector<int>& second_proj) {
if (first_edge == second_edge) {
 return (first_proj == second_proj);
}

bool is_split = (first_edge.first == second_edge.first);

if (!is_split) {
 return true; 
}

if (is_split) {
    Point<int> snd1 = first_edge.second;
    snd1 -= second_edge.second;
    bool pos = 1;
    for(int i = 0; i < d; ++i){
        if(snd1.a[i] < 0) pos = 0;
    }
    if (!pos) {
        return true;
    }
    if (second_proj.size() > first_proj.size()) return false;
    for (int i = 0; i < second_proj.size(); ++i) {
        if (second_proj[i] % 2 != 0 || 
            first_proj[i] != second_proj[i] / 2) {
            return false;
        }
    }
} 
else {
 return true;
}

return true;
}


int n;
vector<Point<double> > points;
vector<Point<int> > mapped_points;
double LENGTH = -1e10;
int L = 0;

void read_input(){
    cin >> n;
    points.resize(n);
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < d; ++j){
            cin >> points[i].a[j];
        }
    }
}

void calc_perturbation() {
    for (auto u : points){
        for(int i = 0; i < d; ++i){
            LENGTH = max(LENGTH, u.a[i]);
        }
    }
    double granularity = LENGTH / (8 * n * c * sqrtd);
    vector<Point<double> > mapped(n);

    for(int i = 0; i < n; ++i){
        auto u = points[i];
        for(int j = 0; j < d; ++j){
            mapped[i].a[j] = round(u.a[j] / granularity) * granularity;
        }
    }

    double adjusted_granularity = LENGTH / (64 * n * c * sqrtd);

    for(int i = 0; i < n; ++i){
        auto u = points[i];
        for(int j = 0; j < d; ++j){
            mapped_points[i].a[j] = round(mapped[i].a[j] / adjusted_granularity);
        }
    }
}

Tree_node* root;

int count_points_in_node(Tree_node* node) {
    int count = 0;
    for (const auto& p : mapped_points) {
        bool inside = true;
        for (int i = 0; i < d; ++i) {
            if (p.a[i] < node->l[i] || p.a[i] >= node->r[i]) {
                inside = false;
                break;
            }
        }
        if (inside) count++;
    }
    return count;
}

int get_point(Tree_node* node) { // returns any point belonging to subdivision
    for (int i = 0; i < mapped_points.size(); ++i) {
        auto p = mapped_points[i];
        bool inside = true;
        for (int i = 0; i < d; ++i) {
            if (p.a[i] < node->l[i] || p.a[i] >= node->r[i]) {
                inside = false;
                break;
            }
        }
        if (inside) return i;
    }
    return -1;
}


vector<Tree_node*> split_node(Tree_node* node, Point<int> shift) {
    vector<Tree_node*> children;
    int split[d];
    for (int i = 0; i < d; ++i) {
        split[i] = (shift.a[i] + L / 2) % L;
    }

    for (int mask = 0; mask < (1 << d); ++mask) {
        Tree_node* child = new Tree_node();
        for (int i = 0; i < d; ++i) {
            if (mask & (1 << i)) {
                child->l[i] = split[i];
                child->r[i] = node->r[i];
            } else {
                child->l[i] = node->l[i];
                child->r[i] = split[i];
            }
        }
        children.push_back(child);
    }
    return children;
}

vector<Tree_node*> split_wrapped_regions(Tree_node* node) {
    vector<Tree_node*> children;
    bool is_wrapped = false;
    for (int i = 0; i < d; ++i) {
        if (node->l[i] >= node->r[i]) {
            is_wrapped = true;
            Tree_node* left = new Tree_node(*node);
            left->r[i] = L;
            vector<Tree_node*> left_children = split_wrapped_regions(left);

            Tree_node* right = new Tree_node(*node);
            right->l[i] = 0;
            vector<Tree_node*> right_children = split_wrapped_regions(right);

            children.insert(children.end(), left_children.begin(), left_children.end());
            children.insert(children.end(), right_children.begin(), right_children.end());
            delete left;
            delete right;
            break;
        }
    }
    if (!is_wrapped) {
        children.push_back(node);
    }
    return children;
}

vector<Tree_node*> regular_split(Tree_node* node) {
    vector<Tree_node*> children;
    int split[d];
    for (int i = 0; i < d; ++i) {
        split[i] = node->l[i] + (node->r[i] - node->l[i]) / 2;
    }

    for (int mask = 0; mask < (1 << d); ++mask) {
        Tree_node* child = new Tree_node();
        for (int i = 0; i < d; ++i) {
            if (mask & (1 << i)) {
                child->l[i] = split[i];
                child->r[i] = node->r[i];
            } else {
                child->l[i] = node->l[i];
                child->r[i] = split[i];
            }
        }
        children.push_back(child);
    }
    return children;
}

void build_quadtree_recursive(Tree_node* node) {
    int cnt = count_points_in_node(node);
    if (cnt == 1){
        node->point_index = get_point(node);
        return;
    }
    if(cnt == 0){
        node->point_index = -1;
        return;
    }

    vector<Tree_node*> children = regular_split(node);
    for (Tree_node* child : children) {
        int child_cnt = count_points_in_node(child);
        if (child_cnt > 0) {
            if (child_cnt >= 1) build_quadtree_recursive(child);
            node->children.push_back(child);
        } else {
            delete child;
        }
    }
}

Tree_node* make_quadtree(Point<int> shift){
    Tree_node* root = new Tree_node();
    for (int i = 0; i < d; ++i) {
        root->l[i] = 0;
        root->r[i] = L;
    }

    vector<Tree_node*> initial_children = split_node(root, shift);
    vector<Tree_node*> processed_children;

    for (Tree_node* child : initial_children) {
        vector<Tree_node*> unwrapped = split_wrapped_regions(child);
        for (Tree_node* uc : unwrapped) {
            int cnt = count_points_in_node(uc);
            if (cnt > 0) {
                if (cnt > 1) build_quadtree_recursive(uc);
                processed_children.push_back(uc);
            } else {
                delete uc;
            }
        }
        delete child;
    }
    root->children = processed_children;
    return root;
}

void setup_possible_child_combinations(const vector<Tree_node*>& children, vector<vector<vector<pair<int,int> > > >& child_confs, vector<vector<pair<int,int> > > & res, int child_idx){
    if(child_idx == children.size()){
        child_confs.push_back(res);
        return;
    }
    for(auto it = children[child_idx]->dp.begin(); it != children[child_idx]->dp.end(); ++it){
        res.push_back(it->first);
        setup_possible_child_combinations(children, child_confs, res, child_idx + 1);
        res.pop_back();
    }
}

void calc_dp(Tree_node* node){
    if(node->children.size() == 0){
        if(node->point_index == -1){
            node->dp[{}] = 0;
            return;
        }else{
            for(auto& conf: possible_configurations){
                auto perm = Permutation(node, &conf);
                vector<Point<int> > portals;
                for(auto u : conf){
                    portals.push_back(perm.get_portal(u.first, u.second));
                }
                if(portals.size() == 0){
                    node->dp[conf] = 0;
                    continue;
                }
                node->dp[conf] = 0;
                long double delta = 1e9;
                for(int i = 0; i < portals.size(); i += 2){
                    if(i == portals.size() - 1){
                        delta = min(delta, dist(portals[i], mapped_points[node->point_index]));
                        continue;
                    }
                    node->dp[conf] += dist(portals[i], portals[i+1]);
                    delta = min(delta, dist(portals[i], mapped_points[node->point_index]) + dist(portals[i + 1], mapped_points[node->point_index]) - dist(portals[i], portals[i+1]));
                }
                node->dp[conf] -= delta;
            }
        }
        return;
    }

    for(int i = 0; i < node->children.size(); ++i){
        calc_dp(node->children[i]);
    }

    for(auto& conf: possible_configurations){
        node->dp[conf] = 1e10;
        auto perm = Permutation(node, &conf);
        vector<vector<vector<pair<int,int> > > > child_confs;
        vector<vector<pair<int,int> > > res;
        setup_possible_child_combinations(node->children, child_confs, res, 0);
        for(auto& comb: child_confs){
            vector<pair<Point<int>, Point<int> > > edges;
            vector<vector<int> > projections;
            for(int i = 0; i < comb.size(); ++i){
                auto perm2 = Permutation(node->children[i], &comb[i]);
                for(int j = 0; j < d * pow2 / 2; ++j){
                    edges.push_back(perm2.get_edge(j));
                    projections.push_back(perm2.get_projection(j));
                }
            }
            bool suitable = 1;
            for(int i = 0; i < edges.size(); ++i){
                for(int j = i + 1; j < edges.size(); ++j){
                    suitable &= Match(edges[i], projections[i], edges[j], projections[j]);
                }
            } 
            if(suitable){
                double val = 0;
                for(int i = 0; i < comb.size(); ++i){
                    val += node->children[i]->dp[comb[i]];
                }
                node->dp[conf] = min(node->dp[conf], val);
            }
        }

    }


}

int main(){
    read_input();
    calc_perturbation();
    m = c * floor(log(n));
    portal_pairs = compute_portal_pairs();
    possible_configurations = precompute_portal_configurations();
    double answ = 1e12;
    for(int I = 0; I < 5; ++I){
        Point<int> shift;
        for(int i = 0; i < d; ++i){
            shift.a[i] = rand() % L;
        }
        root = make_quadtree(shift);
        calc_dp(root);
        answ = min(answ, root->dp[{}]);
    }
    cout << fixed << setprecision(100) << answ * LENGTH / (64 * n * c * sqrtd);
}