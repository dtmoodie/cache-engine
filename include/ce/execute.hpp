#pragma once
#include <ce/CacheEngine.hpp>

// TODO implement cache lookup for static functions
template<class R>
R exec(R(*func)()) {
    void* ptr = (void*)func;
    return func();
}

template<class R, class ... Args>
R exec(R(*func)(Args...), Args...args) {
    return func(args...);
}
