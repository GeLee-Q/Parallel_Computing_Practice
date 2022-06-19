#pragma once

#include <string>

#ifdef _MSC_VER
#ifdef MYLIB_EXPORT
#define MYLIB_API __declspec(dllexport)
#else
#define MYLIB_API __declspec(dllimport)
#endif
#else
#define MYLIB_API
#endif

class MYLIB_API MyLib {
private:
    MyLib() = default;
    MyLib(MyLib const &) = delete;
    MyLib &operator=(MyLib const &) = delete;

    std::string secret;

public:
    static MyLib &getInstance();
    void setSecret(std::string const &in);
    std::string doSomeJobs(std::string const &in) const;
};