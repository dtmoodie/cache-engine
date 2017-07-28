#pragma once
#include <cstddef>
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined ce_EXPORTS
#  define CE_EXPORT __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define CE_EXPORT __attribute__ ((visibility ("default")))
#else
#  define CE_EXPORT
#endif

namespace ce{
    typedef size_t Hash_t;
}