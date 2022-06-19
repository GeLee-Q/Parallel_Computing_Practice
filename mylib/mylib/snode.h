#include <mylib/bate.h>
#include <mutex>
#include <mutex>
#if __has_include(<tbb/spin_mutex.h>)
#include <tbb/spin_mutex.h>
#else  // for students don't have tbb
namespace tbb { using spin_mutex = mutex; };
#endif

#define N (512*512)

template <int Bshift, class Node>
struct DenseBlock {
    static constexpr int numDepth = 0;
    static constexpr bool isPlace = false;
    static constexpr bool bitShift = Bshift;

    static constexpr int B = 1 << Bshift;
    static constexpr int Bmask = B - 1;

    Node m_data[B][B];

    Node *fetch(int x, int y) {
        return &m_data[x & Bmask][y & Bmask];
    }

    Node *touch(int x, int y) {
        return &m_data[x & Bmask][y & Bmask];
    }

    template <class Func>
    void foreach(Func const &func) {
        for (int x = 0; x < B; x++) {
            for (int y = 0; y < B; y++) {
                func(x, y, &m_data[x][y]);
            }
        }
    }
};

template <int Bshift, class Node>
struct PointerBlock {
    static constexpr int numDepth = Node::numDepth + 1;
    static constexpr bool isPlace = false;
    static constexpr bool bitShift = Bshift;

    static constexpr int B = 1 << Bshift;
    static constexpr int Bmask = B - 1;

    std::unique_ptr<Node> m_data[B][B];
    tbb::spin_mutex m_mtx[B][B];

    Node *fetch(int x, int y) {
        return m_data[x & Bmask][y & Bmask].get();
    }

    Node *touch(int x, int y) {
        std::lock_guard _(m_mtx[x & Bmask][y & Bmask]);
        auto &block = m_data[x & Bmask][y & Bmask];
        if (!block)
            block = std::make_unique<Node>();
        return block.get();
    }

    template <class Func>
    void foreach(Func func) {
#pragma omp parallel for collapse(2) firstprivate(func)
        for (int x = 0; x < B; x++) {
            for (int y = 0; y < B; y++) {
                auto ptr = m_data[x][y].get();
                if (ptr)
                    func(x, y, ptr);
            }
        }
    }
};

template <class Node>
struct HashBlock {
    static constexpr int numDepth = Node::numDepth + 1;
    static constexpr bool isPlace = false;
    static constexpr bool bitShift = 0;

    struct MyHash {
        std::size_t operator()(std::tuple<int, int> const &key) const {
            auto const &[x, y] = key;
            return (x * 2718281828) ^ (y * 3141592653);
        }
    };

    std::unordered_map<std::tuple<int, int>, Node, MyHash> m_data;
    tbb::spin_mutex m_mtx;

    Node *fetch(int x, int y) {
        std::lock_guard _(m_mtx);
        auto it = m_data.find(std::make_tuple(x, y));
        if (it == m_data.end())
            return nullptr;
        return &it->second;
    }

    Node *touch(int x, int y) {
        std::lock_guard _(m_mtx);
        auto it = m_data.find(std::make_tuple(x, y));
        if (it == m_data.end()) {
            return &m_data.try_emplace({x, y}).first->second;
        }
        return &it->second;
    }

    template <class Func>
    void foreach(Func func) {
        std::vector<std::tuple<int, int, Node *>> vec;
        for (auto &[key, block]: m_data) {
            auto const &[x, y] = key;
            vec.emplace_back(x, y, &block);
        }
#pragma omp parallel for firstprivate(func)
        for (int i = 0; i < vec.size(); i++) {
            auto const &[x, y, block] = vec[i];
            func(x, y, block);
        }
    }
};

template <class T>
struct PlaceData {
    static constexpr bool isPlace = true;

    T m_value;

    T read() {
        return m_value;
    }

    void write(T value) {
        m_value = value;
    }

    template <class Func>
    void visit(Func const &func) {
        func(m_value);
    }
};

template <class T, class Layout>
struct RootGrid {
    Layout m_root;

    template <class Node>
    static T _read(Node &node, int x, int y) {
        if constexpr (node.isPlace) {
            return node.read();
        } else {
            auto *child = node.fetch(x >> node.bitShift, y >> node.bitShift);
            if (!child)
                return T{};
            return _read(*child, x, y);
        }
    }

    T read(int x, int y) {
        return _read(m_root, x, y);
    }

    struct WriteAccessor {
        Layout &m_root;
        mutable std::array<std::map<std::tuple<int, int>, void *>, Layout::numDepth> cached;

        template <int currDepth, class Node>
        void _write(Node &node, int x, int y, T value) const {
            if constexpr (node.isPlace) {
                return node.write(value);
            } else {
                auto child = [&] {
                    if constexpr (currDepth < Layout::numDepth) {
                        auto &cache = std::get<currDepth>(cached);
                        auto it = cache.find({x >> node.bitShift, y >> node.bitShift});
                        if (it != cache.end()) {
                            return decltype(node.touch(0, 0))(it->second);
                        }
                    }
                    auto *child = node.touch(x >> node.bitShift, y >> node.bitShift);
                    if constexpr (currDepth < Layout::numDepth) {
                        auto &cache = std::get<currDepth>(cached);
                        cache.try_emplace({x >> node.bitShift, y >> node.bitShift}, child);
                    }
                    return child;
                }();
                return _write<currDepth + 1>(*child, x, y, value);
            }
        }

        void write(int x, int y, T value) const {
            return _write<0>(m_root, x, y, value);
        }
    };

    WriteAccessor writeAccess() {
        return {m_root};
    }

    template <class Node, class Func>
    static void _foreach(Node &node, int x, int y, Func const &func) {
        if constexpr (node.isPlace) {
            return node.visit([&] (T &val) {
                func(x, y, val);
            });
        } else {
            int xb = x << node.bitShift;
            int yb = y << node.bitShift;
            return node.foreach([&] (int x, int y, auto *child) {
                _foreach(*child, x, y, func);
            });
        }
    }

    template <class Func>
    void foreach(Func const &func) {
        _foreach(m_root, 0, 0, func);
    }
};

//using ExampleGrid = RootGrid<float, HashBlock<PointerBlock<11, DenseBlock<8, PlaceData<float>>>>>;