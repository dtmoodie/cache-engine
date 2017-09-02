#define CE_DEBUG_CACHE_USAGE
#include <ce/utils.hpp>
#include <ce/CacheEngine.hpp>
#include <ce/sync.hpp>
#include <ce/execute.hpp>
#include <ce/Executor.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>

#ifdef HAVE_OPENCV
#include <ce/cv_sync.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudawarping.hpp>
#endif
#include <unordered_map>

class Allocator: public cv::cuda::GpuMat::Allocator{
public:
    static Allocator* instance(){
        static Allocator g_inst;
        return &g_inst;
    }

    virtual bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize) override{
        if(alloc->allocate(mat, rows, cols, elemSize)){
            hashmap[mat->refcount] = ce::generateHash(ce::generateHash(), rows, cols, elemSize);
            return true;
        }
        return false;
    }

    virtual void free(cv::cuda::GpuMat* mat) override{
        hashmap.erase(mat->refcount);
        alloc->free(mat);
    }

    size_t& getHash(const cv::cuda::GpuMat& mat){
        return hashmap[mat.refcount];
    }

    void setHash(const cv::cuda::GpuMat& mat, size_t hash){
        hashmap[mat.refcount] = hash;
    }
private:
    Allocator(){
        alloc = cv::cuda::GpuMat::defaultAllocator();
        cv::cuda::GpuMat::setDefaultAllocator(this);
    }
    ~Allocator(){
        cv::cuda::GpuMat::setDefaultAllocator(alloc);
    }
    std::unordered_map<int*, size_t> hashmap;
    cv::cuda::GpuMat::Allocator* alloc;
};

namespace ce{
    template<int Idx, class Tuple>
    void saveOutput(size_t hash, Tuple& result, cv::cuda::GpuMat& arg) {
        std::get<Idx>(result) = arg;
        Allocator::instance()->setHash(arg, hash);
    }
    template<int Idx, class Tuple>
    void setOutput(size_t hash, Tuple& result, cv::cuda::GpuMat& arg) {
        arg = std::get<Idx>(result);
        Allocator::instance()->setHash(arg, hash);
    }

    inline size_t generateHash(const cv::cuda::GpuMat& data){
        return Allocator::instance()->getHash(data);
    }
    inline size_t getObjectHash(const cv::cuda::GpuMat& data){
        return Allocator::instance()->getHash(data);
    }
    inline const cv::cuda::GpuMat& getObjectRef(const cv::cuda::GpuMat& data){
        return data;
    }
    inline cv::cuda::GpuMat& getObjectRef(cv::cuda::GpuMat& data) {
        return data;
    }
    namespace type_traits {
        namespace argument_specializations {
            template<class T>
            struct SaveType<cv::OutputArray, T, 2> {
                enum { IS_OUTPUT = 1 };
                typedef typename remove_output<std::remove_reference_t<T>>::type type;
                inline static size_t hash(const T& val) {
                    return 0;
                }
            };
        }
    }

size_t combineHash(size_t seed, const cv::_InputOutputArray& v) {
    (void)v;
    return seed;
}

namespace debug{
    template<class R, class... FArgs, class... Args>
    void debugExecute(R(*func)(FArgs...), Args&&...){
        typedef ce::OutputPack<R(FArgs...), Args...> AI;
        AI::debugPrint();
    }
}
}
int main(int argc, char** argv) {
    Allocator::instance();
	ce::ICacheEngine::setEngine(ce::ICacheEngine::create());
	if (argc == 2){
		cv::Mat h_img = cv::imread(argv[1]);
        cv::cuda::Stream stream1, stream2;
        auto kp = cv::cuda::createGoodFeaturesToTrackDetector(CV_32F);
        auto executor = ce::makeExecutor(*kp);
        cv::cuda::GpuMat output_mat;
        cv::cuda::GpuMat float_mat;
        cv::cuda::GpuMat corners1, corners2;
        auto input = cv::cuda::GpuMat(h_img);
		if (!h_img.empty()) {
            size_t in0 = ce::generateHash(input);
            ce::setCacheUsedLast(false);
            ce::exec(cv::cuda::cvtColor, input, output_mat, cv::COLOR_BGR2GRAY, -1, stream1);
            CV_Assert(ce::wasCacheUsedLast() == false);
            size_t in1 = ce::generateHash(input);
            CV_Assert(in0 == in1);
            size_t out0 = ce::generateHash(output_mat);
            ce::exec(cv::cuda::cvtColor, input, output_mat, cv::COLOR_BGR2GRAY, -1, stream2);
            CV_Assert(ce::wasCacheUsedLast() == true);
            size_t in2 = ce::generateHash(input);
            CV_Assert(in0 == in2);
            size_t out1 = ce::generateHash(output_mat);
            CV_Assert(out0 == out1);
            auto mat_executor = ce::makeExecutor(output_mat);
            cv::cuda::GpuMat float2_mat;

            ce::EXEC_MEMBER(static_cast<void(cv::cuda::GpuMat::*)(cv::OutputArray, int, double, cv::cuda::Stream&)const>(&cv::cuda::GpuMat::convertTo))(
                output_mat, float_mat, CV_32F, 1.0, stream2);
            CV_Assert(ce::wasCacheUsedLast() == false);

            ce::EXEC_MEMBER(static_cast<void(cv::cuda::GpuMat::*)(cv::OutputArray, int, double, cv::cuda::Stream&)const>(&cv::cuda::GpuMat::convertTo))(
                output_mat, float2_mat, CV_32F, 1.0, stream1);

            CV_Assert(ce::wasCacheUsedLast() == true);
            CV_Assert(ce::generateHash(float_mat) == ce::generateHash(float2_mat));
            CV_Assert(ce::generateHash(float_mat) != ce::generateHash(output_mat));

            executor.EXEC_MEMBER(&cv::cuda::CornersDetector::detect)(float_mat, corners1, ce::makeEmptyInput(cv::noArray()), stream2);
            CV_Assert(ce::wasCacheUsedLast() == false);
            executor.EXEC_MEMBER(&cv::cuda::CornersDetector::detect)(float_mat, corners2, ce::makeEmptyInput(cv::noArray()), stream1);
            CV_Assert(ce::wasCacheUsedLast() == true);
            CV_Assert(ce::generateHash(corners1) == ce::generateHash(corners2));
		}
        stream1.waitForCompletion();
        stream2.waitForCompletion();
	}
    
    ce::ICacheEngine::releaseEngine();
	return 0;
}