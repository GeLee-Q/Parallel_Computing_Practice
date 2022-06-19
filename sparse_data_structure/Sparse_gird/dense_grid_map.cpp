#include "bate.h"

#define N (512*512)

struct Grid {
    std::map<std::tuple<int,int>, char> m_data;

    char read(int x, int y) const{
        auto it = m_data.find(std::make_tuple(x, y));
        if(it == m_data.end()){
            return 0;
        }
        return it->second;
    }

    void write(int x, int y, char value){
        m_data[std::make_tuple(x,y)] = value;
    }

    template<class Func>
    void foreach(Func const & func){
        for(auto &[key, value] : m_data){
            auto & [x, y] = key;
            func(x, y , value);
        }
    }
};

int main() {
    bate::timing("main");

    Grid *a = new Grid{};

    float px = 0.f, py = 0.f;
    float vx = 0.2f, vy = 0.6f;

    for (int step = 0; step < N; step++) {
        px += vx;
        py += vy;
        int x = (int)std::floor(px);
        int y = (int)std::floor(py);
        a->write(x, y, 1);
    }

    int count = 0;
    a->foreach([&] (int x, int y, char &value) {
        if (value != 0) {
            count++;
        }
    });
    printf("count: %d\n", count);

    bate::timing("main");
    return 0;
}