cmake_minimum_required(VERSION 3.12)
project(cpp_muduo VERSION 0.0.1 LANGUAGES CXX)


#初始化项目信息，并把当前CMakeLists.txt 所在位置作为根目录
message("PROJECT_NAME: ${PROJECT_NAME}")
message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}") # 当前源码路径 main.cpp
message("PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}") # 当前源码编译输出路径 .exe
message("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message("CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")



#设置cpp版本
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) #如果不支持，直接报错

#告诉编译器不要自动剔除没有引用符号的链接库
set(CMAKE_LINK_WHAT_YOU_USE ON)


#设置构建类型：Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    # set(CMAKE_BUILD_TYPE Debug)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

#不支持MSVC
if(NOT MSVC)
    find_program(CCACHE_PROGRAM ccache)
    if( CCACHE_PROGRAM)
        message(STATUS "Found CCache: ${CCACHE_PROGRAM}")
        set_property(GLOBAL PROPERTY  RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_PROGRAM})
    endif()
endif()

#让静态编译库生成位置无关代码
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#
add_subdirectory(mylib)


#批量添加源代码文件
# CONFIGURE_DEPENDS 自动更新
# GLOB_RECURSE 自动包含所有子文件夹下的文件
file(GLOB_RECURSE main_sources CONFIGURE_DEPENDS src/*.cpp)
add_executable(main ${main_sources})
target_link_libraries(main PRIVATE mylib)



find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found")
endif()

find_package(TBB)
if (TARGET TBB::tbb)
    target_link_libraries(main PUBLIC TBB::tbb)
else()
    message(WARNING "TBB not found")
endif()