#include <iostream>
#include <algorithm>
#include <vector>
#include <random>

using namespace std;

const int d = 2;
const int pow2 = 4;
const int sqrtd = 1;

template<typename T>
struct Point{
    T a[d];
    Point(const T* arr) {
        for(int i = 0; i < d; ++i){
            a[i] = arr[i];
        }
    }
};



const int c = 5;
const double EPS = -10;

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



struct Tree_node{
    int l[d], r[d];
    Tree_node* children[pow2];
};

Tree_node* root;

Tree_node* make_quadtree(int a, int b){

}

int main(){
    read_input();
    calc_perturbation();
    for(int I = 0; I < 5; ++I){
        int a = rand() % L;
        int b = rand() % L;
        root = make_quadtree(a,b);
    }
}