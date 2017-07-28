#include <ce/Executor.hpp>
#include <ce/execute.hpp>
#include <ce/function_traits.hpp>
#include <ce/utils.hpp>

#include <iostream>
#include <memory>
#include <map>
#include <cassert>
using namespace ce;

template<class T, class R, class... Args>
R exec(R(T::*func)(Args...), T* This, Args... args) {
    return (*This.*func)(args...);
}

int foo1(int value){
    return 5;
}

int foo2(int value){
    return 6;
}

int foo3(int value){
    return value;
}

struct bar{
    static int foo(int value){
        return value;
    }
    int member(int value){
        return value;
    }
    void setter(int& value, int& value2){
        value = 5;
        value2 = 10;
    }
    void apply(int value, int& output){
        output = value;
    }
    void member2(int value1, int value2) const{
        
    }
    void overloaded(int ){
    
    }
    void overloaded(float){
    }
    void set(int value){
        m_member = value;
    }
    
    int get(){
        return m_member;
    }
    static int staticFunc(int value){
        return 5;
    }
    int m_member = 0;    
};


// unfortunately we need to use this macro to deal with not being able to hash member function pointers on windows. :(

int main(){
    std::cout << "Testing OutputPack detection\n";
    std::cout << OutputPack<void, int>::OUTPUT_COUNT << " == 0\n";
    std::cout << OutputPack<void, int, int>::OUTPUT_COUNT << " == 0\n";
    std::cout << OutputPack<void, HashedInput<int>, HashedInput<int>>::OUTPUT_COUNT << " == 0\n";
    std::cout << OutputPack<void, HashedOutput<int>>::OUTPUT_COUNT << " == 1\n";
    std::cout << OutputPack<void, HashedOutput<int>, HashedOutput<int>>::OUTPUT_COUNT << " == 2\n";
	

    static_assert(!ce::function_traits::is_const(&bar::member), "member is non const");
    static_assert(ce::function_traits::is_const(&bar::member2), "member is const");

    ICacheEngine::setEngine(std::make_unique<CacheEngine>());
    auto hashed = makeInput<int>(5);
    std::cout << exec(foo1, 0) << std::endl;
    std::cout << exec(foo1, 0) << std::endl;
    std::cout << exec(foo2, 0) << std::endl;
    std::cout << exec(foo3, 10) << std::endl;
    std::cout << exec(bar::foo, 20) << std::endl;
    std::cout << exec(&bar::staticFunc, 0) << std::endl;
    bar cls;
    auto executor = makeExecutor(cls);
    std::cout << executor.EXEC(&bar::member), hashed) << std::endl;
    std::cout << executor.EXEC(&bar::member), hashed) << std::endl;
    int value1 = 20, value2 = 10;
    executor.EXEC(&bar::setter), makeOutput(value1), makeOutput(value2));
    assert(value1 == 5);
    assert(value2 == 10);
    value1 = 100;
    value2 = 200;
    
    executor.EXEC(&bar::setter), makeOutput(value2), makeOutput(value1));
    assert(value1 == 10);
    assert(value2 == 5);
    
    executor.EXEC(&bar::apply), 10, makeOutput(value1));
    executor.EXEC(&bar::apply), 10, makeOutput(value1));
    executor.EXEC(&bar::apply), 5, makeOutput(value1));
    std::cout << "Testing setters and getters" << std::endl;
    int ret = executor.EXEC(&bar::get));
    std::cout << ret << std::endl;
    executor.set(&bar::set, 15);
    std::cout << executor.EXEC(&bar::get)) << std::endl;
    std::cout << executor.EXEC(&bar::get)) << std::endl;

    
    executor.EXEC(&bar::member2), 0,1);
    executor.EXEC(static_cast<void(bar::*)(int)>(&bar::overloaded)), 1);
    executor.EXEC(static_cast<void(bar::*)(float)>(&bar::overloaded)), 1.f);
    
    return 0;
}
