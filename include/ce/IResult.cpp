#include <ce/IResult.hpp>
namespace ce{
IResult::~IResult() {}
size_t IResult::getDynamicSize() const {
    return 0;
}
}

