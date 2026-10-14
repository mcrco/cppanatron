#include "bindings.hpp"

#include <Python.h>

#include "cppanatron_build_config.h"
#include "cppanatron_git.h"

namespace cppanatron::python {

void bind_build_info(nb::module_& module) {
    module.def(
        "build_info",
        [] {
            nb::dict info;
            info["version"] = CPPANATRON_VERSION;
            info["git_sha"] = CPPANATRON_GIT_SHA;
            info["git_dirty"] = CPPANATRON_GIT_DIRTY;
            info["compiler"] = CPPANATRON_COMPILER_ID;
            info["compiler_version"] = CPPANATRON_COMPILER_VERSION;
            info["build_type"] = CPPANATRON_BUILD_TYPE;
            info["cxx_flags"] = CPPANATRON_CXX_FLAGS;
            info["cxx_standard"] = __cplusplus;
            info["nanobind_version"] = nb::str("{}.{}.{}").format(
                NB_VERSION_MAJOR, NB_VERSION_MINOR, NB_VERSION_PATCH);
            info["python_version"] = PY_VERSION;
            return info;
        },
        "How this extension was built: the engine repo's git commit (``git_sha``,\n"
        "\"unknown\" outside a git checkout) and whether tracked files differed from\n"
        "it (``git_dirty``) when it was compiled, the compiler, build type and\n"
        "flags, and the nanobind and Python versions it was built against.");
}

}  // namespace cppanatron::python
