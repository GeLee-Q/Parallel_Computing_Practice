#include <mylib/mylib.h>

MyLib &MyLib::getInstance() {
    static MyLib mylib;
    return mylib;
}

void MyLib::setSecret(std::string const &in) {
    secret = in;
}

std::string MyLib::doSomeJobs(std::string const &in) const {
    return secret + in + secret;
}