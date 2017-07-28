#include <ce/execute.hpp>
#include <ce/Executor.hpp>

struct TestObject{
    TestObject(int init = 0): member1(init){}
    static double staticFunctionWithReturn(){
        return 1.0;
    }

    static void staticReturnRef(double in1, double in2, double& out1, double& out2){
        out1 = in1 + in2;
        out2 = in1 * in2;
    }

    void apply(int in1, int& out1) const{
        out1 = member1 * in1;
    }

    void set(int value){ member1 = value; }

    int get() const{return member1;}

    int apply(int in1, int in2) const{
        return in1 * in2 * member1;
    }

    int member1;
};

struct TestHashedObject: public TestObject {
    ce::Hash_t m_hash = ce::classHash<TestHashedObject>();
};

namespace ce{
    TestHashedObject& getObjectRef(TestHashedObject& obj){
        return obj;
    }
    const TestHashedObject& getObjectRef(const TestHashedObject& obj) {
        return obj;
    }
    Hash_t& getObjectHash(TestHashedObject& obj){
        return obj.m_hash;
    }
    const Hash_t& getObjectHash(const TestHashedObject& obj) {
        return obj.m_hash;
    }
}