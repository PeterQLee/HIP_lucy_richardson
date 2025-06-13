# MIT License
#
# Copyright (c) Peter Q. Lee 2025
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set(libname lucy_richardson_deconv)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${libname} LANGUAGES CXX HIP)
set( CMAKE_BUILD_TYPE Release)
set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")
set(GPU_RUNTIMES "HIP" "CUDA")
set_property(CACHE GPU_RUNTIME PROPERTY STRINGS ${GPU_RUNTIMES})

if(NOT "${GPU_RUNTIME}" IN_LIST GPU_RUNTIMES)
    set(ERROR_MESSAGE
        "GPU_RUNTIME is set to \"${GPU_RUNTIME}\".\nGPU_RUNTIME must be either HIP or CUDA."
    )
    message(FATAL_ERROR ${ERROR_MESSAGE})
endif()

enable_language(${GPU_RUNTIME})
set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)


if(WIN32)
    set(ROCM_ROOT
        "$ENV{HIP_PATH}"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
else()
    set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
endif()


list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")
list(APPEND include_dirs "include/")


find_package(hip REQUIRED)
add_library(${libname} SHARED src/deconv.hip src/gpuOps.hip)

# Make example runnable using ctest

target_link_libraries(${libname} PRIVATE hip::device)
target_include_directories(${libname} PRIVATE ${include_dirs})


add_executable(test tests/test.cc)
target_link_libraries(test PRIVATE ${libname})
target_include_directories(test PRIVATE ${include_dirs})

# Copy test files
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tests/rgbimg.bin DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tests/rgbrestored.bin DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tests/impulse.bin DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(LR_GPU_wrapper src/wrapper_py.cc)
target_link_libraries(LR_GPU_wrapper PRIVATE ${libname})
target_include_directories(LR_GPU_wrapper PRIVATE ${include_dirs})

install(TARGETS ${example_name})
