cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(inference_cfile)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${OPENCV_CXX_FLAGS}")

add_executable(inference_cfile inference_cfile.cpp)
target_link_libraries(inference_cfile "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET inference_cfile PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to http://github.com/pytorch/pytorch/issues/25457,
# The DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET inference_cfile
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TORCH_DLLS}
                        $<TARGET_FILE_DIR:inference_cfile>)
endif (MSVC)