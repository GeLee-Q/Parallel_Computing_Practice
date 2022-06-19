#include "bate.h"

#define N (512*512)

struct Grid {
    struct MyHash {
        std::size_t operator()(std::tuple<int, int> const &key) const {
            auto const &[x, y] = key;
            return (x * 2718281828) ^ (y * 3141592653);
        }
    };

    
    static constexpr int Bshift = 4;
    static constexpr int B = 1 << Bshift;
    static constexpr int Bmask = B - 1;


    // 扁平化
    struct Block {
        char m_block[B * B];
    };

    std::unordered_map<std::tuple<int, int>, Block, MyHash> m_data;  // ~1MB


    // x / B == x << Bshift ， << 可以变成 & 和 ~ 
    // x % B == x & (B - 1)
    // x 
    char read(int x, int y) const {
        auto it = m_data.find(std::make_tuple(x & ~Bmask, y & ~Bmask));
        if (it == m_data.end()) {
            return 0;
        }
        return it->second.m_block[((x & Bmask) << Bshift) | (y & Bmask)];
    }

    void write(int x, int y, char value) {
        Block &block = m_data[std::make_tuple(x & ~Bmask, y & ~Bmask)];
        block.m_block[((x & Bmask) << Bshift) | (y & Bmask)] = value;
    }

    template <class Func>
    void foreach(Func const &func) {
        for (auto &[key, block]: m_data) {
            auto &[xb, yb] = key;
            for (int dx = 0; dx < B; dx++) {
                for (int dy = 0; dy < B; dy++) {
                    func(xb | dx, yb | dy, block.m_block[(dx << Bshift) | dy]);
                }
            }
        }
    }
};

int main() {
    bate::timing("main");

    Grid *a = new Grid{};

    float px = -100.f, py = 100.f;
    float vx = 0.2f, vy = -0.6f;

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