#include "bate.h"

#define N (512*512)

// How to write your hash
// struct hashfunc
// {
//     template<typename T, typename U>
//     size_t operator() (const pair<T, U> &i) const
//     {
//         return hash<T>()(i.first) ^ hash<U>()(i.second);
//     }
// };
// unordered_map< pair<int, int>, int , hashfunc > mp;


struct Grid {
    struct MyHash{
        std::size_t operator()(std::tuple<int,int> const & key) const{
            auto const & [x,y] = key;
            return (x * 2718281828 ) ^ (y* 31415926538);
        }
    };

    static constexpr int B = 16;
    struct Block{
        char m_block[B][B];
    };

    std::unordered_map<std::tuple<int,int>, Block, MyHash> m_data;
    //映射 位置----> 数值块
    // x / B , y / B 可以确定是哪一个块
    // x % B , y % B 则可以确定是块中的哪一个位置


    char read(int x, int y) const{
        auto it = m_data.find(std::make_tuple(x / B, y / B));
        if(it == m_data.end()){
            return 0;
        }
        return it->second.m_block[x % B][y % B];
    }

    void write(int x, int y, char value){
        Block & block = m_data[std::make_tuple(x / B, y / B)];
        block.m_block[ x % B][ y % B] = value;
    }

    template<class Func>
    void foreach(Func const & func){
        for(auto &[key, block] : m_data){
            //得到 块的信息
            auto & [xb, yb] = key;
            // 需要对每一个块做的循环；
            for(int dx = 0 ; dx < B; dx++){
                for(int dy = 0 ; dy < B; dy++){
                    int _x = xb * B + dx;
                    int _y = yb * B + dy;
                    func(_x, _y , block.m_block[dx][dy]);
                }
            }
            
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