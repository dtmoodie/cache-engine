struct MutateOutputObject;
#include <cstring>

namespace ce
{
    const size_t& getObjectHash(const MutateOutputObject& obj);
    size_t& getObjectHash(MutateOutputObject& obj);
    MutateOutputObject& get(MutateOutputObject& obj);
    const MutateOutputObject& get(const MutateOutputObject& obj);
} // namespace ce

#include <ce/Executor.hpp>
#include <ce/execute.hpp>

struct TestObject
{
    TestObject(int init = 0)
        : member1(init)
    {
    }
    static double staticFunctionWithReturn()
    {
        return 1.0;
    }

    static void staticReturnRef(double in1, double in2, double& out1, double& out2)
    {
        out1 = in1 + in2;
        out2 = in1 * in2;
    }

    void apply(int in1, int& out1) const
    {
        out1 = member1 * in1;
    }

    void set(int value)
    {
        member1 = value;
    }

    int get() const
    {
        return member1;
    }

    int apply(int in1, int in2) const
    {
        return in1 * in2 * member1;
    }

    int member1;
};

struct TestOutputObject
{
    int data;
};

struct TestHashedOutputObject : public TestOutputObject, ce::HashedBase
{
};

struct MutateOutputObject
{
    void mutate(TestOutputObject& obj)
    {
        obj.data = member;
    }
    void set(int val)
    {
        member = val;
    }

  protected:
    friend size_t& ce::getObjectHash(MutateOutputObject& obj);
    friend const size_t& ce::getObjectHash(const MutateOutputObject& obj);
    size_t hash = 0;
    int member;
};

struct TestHashedObject : public TestObject
{
    size_t m_hash = 0;
};

namespace ce
{
    TestHashedObject& get(TestHashedObject& obj)
    {
        return obj;
    }

    const TestHashedObject& get(const TestHashedObject& obj)
    {
        return obj;
    }

    size_t& getObjectHash(TestHashedObject& obj)
    {
        return obj.m_hash;
    }

    const size_t& getObjectHash(const TestHashedObject& obj)
    {
        return obj.m_hash;
    }

    size_t& getObjectHash(MutateOutputObject& obj)
    {
        return obj.hash;
    }
    const size_t& getObjectHash(const MutateOutputObject& obj)
    {
        return obj.hash;
    }
    MutateOutputObject& get(MutateOutputObject& obj)
    {
        return obj;
    }
    const MutateOutputObject& get(const MutateOutputObject& obj)
    {
        return obj;
    }
} // namespace ce
