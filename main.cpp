#include <ct/String.hpp>
#include <iostream>
#include <memory>
#include <map>
#include <cassert>
int foo1(){
    return 5;
}

int foo2(){
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
    
    int m_member = 0;    
};

template<class R>
R exec(R(*func)()){
    return func();
}

template<class R, class ... Args>
R exec(R(*func)(Args...), Args...args){
    return func(args...);
}

template<class T, class R, class... Args>
R exec(R(T::*func)(Args...),T* This, Args... args){
    return (*This.*func)(args...);
}

struct IResult{
    virtual ~IResult();
};

IResult::~IResult(){}

template<class ... T> struct TResult: IResult{
    TResult(std::tuple<T...>&& arg):
        values(std::move(arg)){}
    TResult(T&&... args):
    values(std::forward<T>(args)...){
        
    }

    std::tuple<T...> values;
};

template<class...T> struct TResult<std::tuple<T...>>: IResult{
    TResult(std::tuple<T...>&& arg):
        values(std::move(arg)){}
    TResult(T&&... args):
    values(std::forward<T>(args)...){
        
    }

    std::tuple<T...> values;
};

struct CacheEngine{
    static CacheEngine& instance(){
        static CacheEngine inst;
        return inst;
    }

    std::shared_ptr<IResult>& getCachedResult(size_t hash){
        return m_result_cache[hash];
    }
private:
    CacheEngine(){}
    std::map<size_t, std::shared_ptr<IResult>> m_result_cache;
};

// lolol poormans hash for now
size_t generateHash(){
    static size_t count = 0;
    return ++count;
}

template<class T> struct HashedInput{
    template<class...Args> HashedInput(Args&&... args):
        data(std::forward<Args>(args)...){
        hash = generateHash();
    }

    size_t hash;
    T data;
};


template<class T> struct HashedOutput{
    HashedOutput(HashedOutput<T>&& other):
        m_ref(other.m_ref){
        
    }

    HashedOutput(HashedOutput<T>& other):
    m_ref(other.m_ref){
        
    }

    HashedOutput(T& ref):
        m_ref(ref){}
    
    T& m_ref;
};

template<class T> T& get(T& data){
    return data;
}

template<class T> T&& get(T&& data){
    return std::forward<T>(data);
}

template<class T> T& get(HashedInput<T>& data){
    return data.data;
}

template<class T> T& get(HashedOutput<T>& data){
    return data.m_ref;
}

template<class T> T& get(HashedOutput<T>&& data){
    return data.m_ref;
}

template<class T>
std::size_t combineHash(std::size_t seed, const T& v){
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    return seed;
}

std::size_t combineHash(std::size_t seed, std::size_t hash){
    seed ^= hash + 0x9e3779b9 + (seed<<6) + (seed>>2);
    return seed;
}

template<class T> 
std::size_t combineHash(std::size_t seed, const HashedInput<T>& v){
    seed ^= v.hash + 0x9e3779b9 + (seed<<6) + (seed>>2);
    return seed;
}

template<class T>
std::size_t combineHash(std::size_t seed, const HashedOutput<T>& v){
    (void)v;
    return seed;
}

template<class T> 
size_t generateHash(std::size_t seed, T&& v){
    return combineHash(seed, std::forward<T>(v));
}
std::size_t generateHash(std::size_t seed){
    return seed;
}

template<class T, class R, class... FArgs>
std::size_t generateHash(R(T::*func)(FArgs...)){
    //return reinterpret_cast<std::size_t>((void*)func);
    std::hash<R(T::*)(FArgs...)> hasher;
    return hasher(func);
}

template<class T, class...Args>
size_t generateHash(std::size_t seed, T&& v, Args&&... args){
    return generateHash(combineHash(seed, std::forward<T>(v)), std::forward<Args>(args)...);
}

template<typename... Args>
struct variadic_typedef {};

template<typename... Args>
struct convert_in_tuple {
    //Leaving this empty will cause the compiler
    //to complain if you try to access a "type" member.
    //You may also be able to do something like:
    //static_assert(std::is_same<>::value, "blah")
    //if you know something about the types.
};

template<typename... Args>
struct convert_in_tuple< variadic_typedef<Args...> > {
    //use Args normally
    typedef std::tuple<Args...> type;
};

template<typename T, class T2>
struct append_to_tupple{};

template<typename T, typename... Args>
struct append_to_tupple<T, variadic_typedef<Args...> > {
    typedef variadic_typedef<T, Args...> type;
    typedef std::tuple<T, Args...> tuple_type;
};

template<class Enable, class T, class...Args> struct OutputPack: public OutputPack<void, Args...>{
    enum {
        OUTPUT_COUNT = OutputPack<void, Args...>::OUTPUT_COUNT
    };
};

template<class T> struct OutputPack<void, T>{
    enum{
        OUTPUT_COUNT = 0
    };
};

template<class T> struct OutputPack<void, HashedOutput<T>>{
    enum{
        OUTPUT_COUNT = 1
    };
    typedef variadic_typedef<T> types;
    
    template<class TupleType> 
    static void setOutputs(TupleType& result, HashedOutput<T>& out){
        get(out) = std::get<std::tuple_size<TupleType>::value - 1>(result);
    }
    template<class TupleType> 
    static void saveOutputs(TupleType& result, HashedOutput<T>& out){
        std::get<std::tuple_size<TupleType>::value - 1>(result) = get(out);
    }
};

template<class T, class ... Args> struct OutputPack<typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT != 0>::type, HashedOutput<T>, Args...>: public OutputPack<void, Args...>{
    enum {
        OUTPUT_COUNT = OutputPack<void, Args...>::OUTPUT_COUNT + 1
    };
    typedef typename append_to_tupple<T, typename OutputPack<void, Args...>::types>::type types;
    
    template<typename TupleType>
    static void setOutputs(TupleType& result, HashedOutput<T>& out, Args&... args){
        get(out) = std::get<std::tuple_size<TupleType>::value - OUTPUT_COUNT >(result);
        OutputPack<void, Args...>::setOutputs(result, args...);
    }
    
    template<typename TupleType>
    static void saveOutputs(TupleType& result, HashedOutput<T>& out, Args&... args){
        std::get<std::tuple_size<TupleType>::value - OUTPUT_COUNT >(result) = get(out);
        OutputPack<void, Args...>::saveOutputs(result, args...);
    }
};

template<class T, class ... Args> 
struct OutputPack<typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT == 0>::type, HashedOutput<T>, Args...>: public OutputPack<void, Args...>{
    enum {
        OUTPUT_COUNT = OutputPack<void, Args...>::OUTPUT_COUNT + 1
    };
    typedef variadic_typedef<T> types;
    
    template<class TupleType>
    static void setOutputs(TupleType& result, HashedOutput<T>& out, Args&... args){
        get(out) = std::get<std::tuple_size<TupleType>::value - OutputPack<void, Args...>::OUTPUT_COUNT - 1>(result);
        OutputPack<void, Args...>::setOutputs(result, args...);
    }
    
    template<class TupleType>
    static void saveOutputs(TupleType& result, HashedOutput<T>& out, Args&... args){
        std::get<std::tuple_size<TupleType>::value - OutputPack<void, Args...>::OUTPUT_COUNT - 1>(result) = get(out);
        OutputPack<void, Args...>::saveOutputs(result, args...);
    }
};
#define CE_MEM_FUN(fun) #fun, fun

template<class T, class ... Args> 
struct OutputPack<typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT != 0>::type, T, Args...>: public OutputPack<void, Args...>{
    enum {
        OUTPUT_COUNT = OutputPack<void, Args...>::OUTPUT_COUNT
    };
    typedef typename OutputPack<void, Args...>::types types;
    typedef typename convert_in_tuple<types>::type TupleType;
    
    static void setOutputs(TupleType& result, T&, Args&... args){
        OutputPack<void, Args...>::setOutputs(result, args...);
    }
    static void saveOutputs(TupleType& result, T&, Args&... args){
        OutputPack<void, Args...>::saveOutputs(result, args...);
    }
};

template<class T, class ... Args> 
struct OutputPack<typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT == 0>::type, T, Args...>: public OutputPack<void, Args...>{
    enum {
        OUTPUT_COUNT = OutputPack<void, Args...>::OUTPUT_COUNT
    };
};


template<class T> struct Executor{    
    Executor(T& obj):m_obj(obj){}

    template<uint32_t fhash, class R, class...FArgs, class... Args>
    R exec(R(T::*func)(FArgs...), Args&&... args){
        size_t hash = generateHash(m_hash, args...);
        hash = combineHash(hash, fhash);
        std::cout << "Hash : " << hash << std::endl;
        std::shared_ptr<IResult>& result = CacheEngine::instance().getCachedResult(hash);
        if(result){
            std::shared_ptr<TResult<R>> tresult = std::dynamic_pointer_cast<TResult<R>>(result);
            if(tresult){
                std::cout << "Found result in cache" << std::endl;
                return std::get<0>(tresult->values);
            }
        }
        
        R ret = (m_obj.*func)(get(std::forward<Args>(args))...);
        result.reset(new TResult<R>(std::forward<R>(ret)));
        return ret;
    }

    template<uint32_t fhash, class...FArgs, class... Args>
    typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT == 0>::type exec(void(T::*func)(FArgs...) const, Args&&... args) {
        // no output but it's a const call soooo execute
        (m_obj.*func)(get(std::forward<Args>(args))...);
    }

    template<uint32_t fhash, class...FArgs, class... Args>
    typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT == 0>::type exec(void(T::*func)(FArgs...), Args&&... args) {
        // Assuming this modifies the object since there is no output
        size_t hash = generateHash(m_hash, args...);
        m_hash = combineHash(hash, fhash);
        (m_obj.*func)(get(std::forward<Args>(args))...);
    }

    template<uint32_t fhash, class...FArgs, class...Args>
    typename std::enable_if<OutputPack<void,Args...>::OUTPUT_COUNT != 0>::type exec(void(T::*func)(FArgs...), Args&&... args){
        typedef OutputPack<void,Args...> PackType;
        typedef typename convert_in_tuple<typename PackType::types>::type output_tuple_type;
        size_t hash = generateHash(m_hash, args...);
        hash = combineHash(hash, fhash);
        std::cout << "Hash: " << hash << std::endl;
        std::shared_ptr<IResult>& result = CacheEngine::instance().getCachedResult(hash);
        if(result){
            std::shared_ptr<TResult<output_tuple_type>> tresult = std::dynamic_pointer_cast<TResult<output_tuple_type>>(result);
            if(tresult){
                std::cout << "Found result in cache" << std::endl;
                PackType::setOutputs(tresult->values, args...);
                return;
            }
        }
        (m_obj.*func)(get(std::forward<Args>(args))...);
        output_tuple_type results;
        PackType::saveOutputs(results, args...);
        result.reset(new TResult<output_tuple_type>(std::move(results)));
    }
    
    template<class... Args>
    void set(void(T::*func)(Args...), Args&&...args){
        m_hash = generateHash(m_hash, args...);
        (m_obj.*func)(get(std::forward<Args>(args))...);
    }

    T& m_obj;
    std::size_t m_hash = generateHash();
};

template<class T>
Executor<T> make_executor(T& obj){
    return Executor<T>(obj);
}

template<class T, class... Args> HashedInput<T> make_input(Args&&... args){
    return HashedInput<T>(std::forward<Args>(args)...);
}

template<class T> HashedOutput<T> make_output(T& ref){
    return HashedOutput<T>(ref);
}
#define EXEC(func) exec<ct::ctcrc32(#func)>(func

int main(){
    std::cout << "Testing OutputPack detection\n";
    std::cout << OutputPack<void, int>::OUTPUT_COUNT << " == 0\n";
    std::cout << OutputPack<void, int, int>::OUTPUT_COUNT << " == 0\n";
    std::cout << OutputPack<void, HashedInput<int>, HashedInput<int>>::OUTPUT_COUNT << " == 0\n";
    std::cout << OutputPack<void, HashedOutput<int>>::OUTPUT_COUNT << " == 1\n";
    std::cout << OutputPack<void, HashedOutput<int>, HashedOutput<int>>::OUTPUT_COUNT << " == 2\n";

    CacheEngine& ce = CacheEngine::instance();
    (void)ce;
    auto hashed = make_input<int>(5);
    std::cout << exec(foo1) << std::endl;
    std::cout << exec(foo2) << std::endl;
    std::cout << exec(foo3, 10) << std::endl;
    std::cout << exec(bar::foo, 20) << std::endl;
    bar cls;
    auto executor = make_executor(cls);
    std::cout << executor.EXEC(&bar::member), hashed) << std::endl;
    std::cout << executor.EXEC(&bar::member), hashed) << std::endl;
    int value1 = 20, value2 = 10;
    executor.EXEC(&bar::setter), make_output(value1), make_output(value2));
    assert(value1 == 5);
    assert(value2 == 10);
    value1 = 100;
    value2 = 200;
    
    executor.EXEC(&bar::setter), make_output(value2), make_output(value1));
    assert(value1 == 10);
    assert(value2 == 5);
    
    executor.EXEC(&bar::apply), 10, make_output(value1));
    executor.EXEC(&bar::apply), 10, make_output(value1));
    executor.EXEC(&bar::apply), 5, make_output(value1));
    std::cout << "Testing setters and getters" << std::endl;
    int ret = executor.EXEC(&bar::get));
    std::cout << ret << std::endl;
    executor.set(&bar::set, 15);
    std::cout << executor.EXEC(&bar::get)) << std::endl;
    std::cout << executor.EXEC(&bar::get)) << std::endl;

    
    executor.EXEC(&bar::member2), 0,1);
    executor.EXEC(static_cast<void(bar::*)(int)>(&bar::overloaded)), 1);
    executor.EXEC(static_cast<void(bar::*)(float)>(&bar::overloaded)), 1.f);
    //std::cout << exec(&bar::member, &cls, 11) << std::endl;
    return 0;
}
