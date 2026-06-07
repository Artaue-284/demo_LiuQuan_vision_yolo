cmake_minimum_required(VERSION 3.16)
project(RMArmorDetect VERSION 1.0.0 LANGUAGES CXX)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
#  Option: use ONNX Runtime instead of OpenCV DNN
#    cmake .. -DUSE_ONNX_RUNTIME=ON -DONNX_RUNTIME_DIR=...
# ============================================================
option(USE_ONNX_RUNTIME "Use ONNX Runtime C++ API instead of OpenCV DNN" OFF)

# ============================================================
#  1) OpenCV (required for both paths)
# ============================================================
find_package(OpenCV REQUIRED COMPONENTS core imgproc video dnn highgui calib3d)
message(STATUS "OpenCV  version: ${OpenCV_VERSION}")
message(STATUS "OpenCV  libs:    ${OpenCV_LIBS}")
message(STATUS "OpenCV  include: ${OpenCV_INCLUDE_DIRS}")

# ============================================================
#  2) Optional ONNX Runtime
# ============================================================
if(USE_ONNX_RUNTIME)
    if(NOT DEFINED ONNX_RUNTIME_DIR)
        message(FATAL_ERROR "USE_ONNX_RUNTIME=ON requires -DONNX_RUNTIME_DIR=<path>")
    endif()
    find_path(ONNX_RUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
              HINTS "${ONNX_RUNTIME_DIR}/include")
    find_library(ONNX_RUNTIME_LIBRARY onnxruntime
                 HINTS "${ONNX_RUNTIME_DIR}/lib")
    if(NOT ONNX_RUNTIME_INCLUDE_DIR OR NOT ONNX_RUNTIME_LIBRARY)
        message(FATAL_ERROR "ONNX Runtime not found under ${ONNX_RUNTIME_DIR}")
    endif()
    message(STATUS "ONNX Runtime include: ${ONNX_RUNTIME_INCLUDE_DIR}")
    message(STATUS "ONNX Runtime lib:     ${ONNX_RUNTIME_LIBRARY}")
    add_compile_definitions(USE_ONNX_RUNTIME)
endif()

# ============================================================
#  Sources
# ============================================================
set(SOURCES
    src/main.cpp
    src/Detector.cpp
    src/KalmanFilter.cpp
    src/Tracker.cpp
    src/PnPSolver.cpp
)

set(HEADERS
    include/Detector.h
    include/KalmanFilter.h
    include/Tracker.h
    include/PnPSolver.h
)

# ============================================================
#  Executable
# ============================================================
add_executable(rm_armor_detect ${SOURCES} ${HEADERS})

target_include_directories(rm_armor_detect PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(rm_armor_detect PRIVATE ${OpenCV_LIBS})

if(USE_ONNX_RUNTIME)
    target_include_directories(rm_armor_detect PRIVATE ${ONNX_RUNTIME_INCLUDE_DIR})
    target_link_libraries(rm_armor_detect PRIVATE ${ONNX_RUNTIME_LIBRARY})
endif()

# ============================================================
#  Platform-specific
# ============================================================
if(WIN32)
    # Copy OpenCV DLLs to output directory on Windows
    if(MSVC)
        set_target_properties(rm_armor_detect PROPERTIES
            VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
        )
    endif()

    # Try to find and copy ONNX Runtime DLL
    if(USE_ONNX_RUNTIME AND EXISTS "${ONNX_RUNTIME_DIR}/bin")
        file(GLOB ONNX_DLLS "${ONNX_RUNTIME_DIR}/bin/*.dll")
        foreach(DLL ${ONNX_DLLS})
            add_custom_command(TARGET rm_armor_detect POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DLL}" "$<TARGET_FILE_DIR:rm_armor_detect>"
                COMMENT "Copying ${DLL}")
        endforeach()
    endif()
endif()

# ============================================================
#  Install
# ============================================================
install(TARGETS rm_armor_detect DESTINATION bin)

message(STATUS "Build configuration done.")
message(STATUS "  USE_ONNX_RUNTIME = ${USE_ONNX_RUNTIME}")
