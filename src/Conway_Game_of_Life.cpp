#include <cstdio>
#include <vector>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <memory>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <tbb/spin_mutex.h>
#include <mutex>
#include <atomic>

#include <mylib/ticktock.h>
#include <mylib/bate.h>


struct Grid {
    static constexpr int Bshift = 8;
    static constexpr int B = 1 << Bshift;
    static constexpr int Bmask = B - 1;

    static constexpr int B1shift = 11;
    static constexpr int B1 = 1 << B1shift;
    static constexpr int B1mask = B1 - 1;

    struct Block {
        char m_block[B][B];
    };

    tbb::spin_mutex m_mtx[B1][B1];
    std::unique_ptr<Block> m_data[B1][B1];  // ~1MB

    char read(int x, int y) const {
        auto &block = m_data[(x >> Bshift) & B1mask][(y >> Bshift) & B1mask];
        if (!block)
            return 0;
        return block->m_block[x & Bmask][y & Bmask];
    }

    void write(int x, int y, char value) {
        auto &block = m_data[(x >> Bshift) & B1mask][(y >> Bshift) & B1mask];
        if (!block) {
            std::lock_guard _(m_mtx[(x >> Bshift) & B1mask][(y >> Bshift) & B1mask]);
            if (!block)
                block = std::make_unique<Block>();
        }
        block->m_block[x & Bmask][y & Bmask] = value;
    }

    template <class Func>
    void foreach(Func const &func) {
#pragma omp parallel for collapse(2)
        for (int x1 = 0; x1 < B1; x1++) {
            for (int y1 = 0; y1 < B1; y1++) {
                auto const &block = m_data[x1 & B1mask][y1 & B1mask];
                if (!block)
                    continue;
                int xb = x1 << B1shift;
                int yb = y1 << B1shift;
                for (int dx = 0; dx < B; dx++) {
                    for (int dy = 0; dy < B; dy++) {
                        func(xb | dx, yb | dy, block->m_block[dx][dy]);
                    }
                }
            }
        }
    }
};

constexpr int N = 1<<12;

// 任务: 改造稀疏数据结构
// std::vector<bool> cells(N * N);
// std::vector<bool> outcells(N * N);
Grid * cells = new Grid{};
Grid * outcells = new Grid{};


const char gundata[] = R"(125b3o$125bobo$125b3o6bo$125b3o5b3o$125b3o$125b3o$125bobo5b3o$125b3o$
133bobo$133bobo2$133b3o$111b2o6b2o$110bo2bo4bo2bo$110bo2bo4bo2bo11b3o$
110bo2bo4bo2bo12bo$111b2o6b2o2$137bo2b2o4b2o2bo$136bo3b3o2b3o3bo$137bo
2b2o4b2o2bo7$139bo$139b3o4bobo2bobo$138bob4obo2bo2bo2bob2o$120bobo23bo
bo2bobo$120b2o$121bo$139b3o$139b3o$44b3o93bo$45bo94bo$45bo94bo$44b3o
92bobo2$44b3o$44b3o92bobo$140bo$44b3o93bo$45bo94bo$45bo93b3o$44b3o92b
3o2$51b2o6b2o$49bo4bo2bo4bo$49bo4bo2bo4bo$49bo4bo2bo4bo4bo$51b2o6b2o5b
3o8bo$65bobobo6b3o$65bobobo5bobobo$66b3o6bobobo$67bo8b3o$77bo2$67bo$
66b3o8bo$17bo47bobobo6b3o$16b3o46bobobo5bobobo$15bobobo46b3o6bobobo$
15bobobo47bo8b3o$16b3o12bo45bo$17bo11b2o$30b2o2$17bo79b3o$16b3o80bo$
15bobobo59bo18bo$15bobobo59bobo$16b3o49bo10b2o$17bo49b2o$67bobo$78b3o$
2bo2bo4bo2bo64bo$3o2b6o2b3o63bo$2bo2bo4bo2bo2$81b2o$82b2o$81bo2$48bo$
46b2o$47b2o$58bobo2bobo$54b2obo2bo2bo2bob2o$58bobo2bobo2$22b3o52b3o2$
21bo3bo50bo3bo$21bo3bo50bo3bo2$22b3o52b3o$88b3o$88b3o$22b3o52b3o9bo$
89bo$21bo3bo50bo3bo8bo$21bo3bo17bobo30bo3bo7bobo$44b2o$22b3o19bo32b3o$
88bobo$89bo$8bo2bob2obo2bo69bo$8b4ob2ob4o69bo$8bo2bob2obo2bo68b3o$37b
2o6b2o41b3o$35bo4bo2bo4bo$35bo4bo2bo4bo$35bo4bo2bo4bo$37b2o6b2o2$51b3o
66b2o$52bo66b2o$52bo68bo$51b3o2$51b3o$38bobo10b3o$39b2o$39bo11b3o$52bo
$52bo$51b3o26$159bo$159b3o4bobo2bobo$158bob4obo2bo2bo2bob2o$166bobo2bo
bo3$159b3o$159b3o$160bo$160bo$81bo78bo$79bobo77bobo$80b2o2$159bobo$
160bo$160bo$160bo$159b3o$159b3o23$117b3o$119bo$118bo$100b2o6b2o$99bo2b
o4bo2bo$99bo2bo4bo2bo$99bo2bo4bo2bo$100b2o6b2o$114b3o2$113bo3bo$113bo
3bo2$114b3o3$114b3o2$113bo3bo$113bo3bo2$114b3o!)";

void init(int bx, int by) {

    int acc = 0;
    int x = bx;
    int y = by;
    for (int i = 0; i < sizeof(gundata); i++) {
        char c = gundata[i];
        if (!c || strchr(" \n\t!", c)) continue;
        if ('0' <= c && c <= '9') {
            acc *= 10;
            acc += c - '0';
            continue;
        }
        if (!acc) acc = 1;
        if (c == 'b') {
            for (int o = 0; o < acc; o++) {
               cells->write(x, y++, 0);
            }
        }
        if (c == 'o') {
            for (int o = 0; o < acc; o++) {
                cells->write(x, y++, 1);
            }
        }
        if (c == '$') {
            y = by;
            x += acc;
        }
        acc = 0;
    }
}


void step() {
   
    tbb::parallel_for(tbb::blocked_range2d<int>(1,N-1,1,N-1),
    [&](tbb::blocked_range2d<int> r){
        for(int y = r.cols().begin() ; y < r.cols().end(); y++){
            for(int x = r.rows().begin(); x < r.rows().end(); x++){
                int neigh = 0;
            
            neigh += cells->read(x , y+1);
            neigh += cells->read(x , y-1);

            neigh += cells->read((x + 1) , (y + 1));
            neigh += cells->read((x + 1) , y);
            neigh += cells->read((x + 1) , y-1);
            neigh += cells->read((x - 1) , y+1);
            neigh += cells->read((x - 1) , y);
            neigh += cells->read((x - 1) , y-1);
            
            if (cells->read(x , y)) {
                if (neigh == 2 || neigh == 3) {
                    outcells->write(x, y, 1);
                   
                } else {
                    outcells->write(x, y, 0);
                    
                }
            } else {
                if (neigh == 3) {
                    outcells->write(x, y, 1);
                    
                } else {
                    outcells->write(x, y, 0);
                    
                }
            }

            }
        }
    });
    std::swap(cells, outcells);
}

void step1() {
#pragma omp parallel for collapse(2)
    for (int y = 1; y < N-1; y++) {
        for (int x = 1; x < N-1; x++) {
            int neigh = 0;
            
            neigh += cells->read(x , y+1);
            neigh += cells->read(x , y-1);

            neigh += cells->read((x + 1) , (y + 1));
            neigh += cells->read((x + 1) , y);
            neigh += cells->read((x + 1) , y-1);
            neigh += cells->read((x - 1) , y+1);
            neigh += cells->read((x - 1) , y);
            neigh += cells->read((x - 1) , y-1);
            
            if (cells->read(x , y)) {
                if (neigh == 2 || neigh == 3) {
                    outcells->write(x, y, 1);
                } else {
                    outcells->write(x, y, 0);
                }
            } else {
                if (neigh == 3) {
                    outcells->write(x, y, 1);
                } else {
                    outcells->write(x, y, 0);
                }
            }
        }
    }
    std::swap(cells, outcells);


}



void showinfo() {
    int rightbound = std::numeric_limits<int>::min();
    int leftbound = std::numeric_limits<int>::max();
    int count = 0;
#pragma omp parallel for collapse(2) reduction(max:rightbound) reduction(min:leftbound) reduction(+:count)
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            if (cells->read(x , y)) {
                rightbound = std::max(rightbound, y);
                leftbound = std::min(leftbound, y);
                count++;
            }
        }
    }
    // 标准答案：left=1048, right=3523, count=2910
    printf("left=%d, right=%d, count=%d\n", leftbound, rightbound, count);
}

int main() {
    TICK(main);

    init(N / 2, N / 2);
    init(N / 2 - 500, N / 2 - 500);
    init(N / 2 + 500, N / 2 + 500);
    init(N / 2 - 1000, N / 2 - 1000);
    init(N / 2 + 1000, N / 2 + 1000);
    printf("init is ok\n");
    for (int times = 0; times < 800; times++) {
        printf("step %d\n", times);
        if (times % 100 == 0)
            showinfo();
        
        step();
    }
    showinfo();

    TOCK(main);
    return 0;
}