# no cuda
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# cuda required
#cmake_minimum_required(VERSION 3.13.2 FATAL_ERROR)

message(STATUS "CMake version: ${CMAKE_VERSION}")

project(math21)

set(CMAKE_CXX_STANDARD 11)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wvla")
    #    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pedantic")
    message(STATUS "CMAKE_CXX_FLAGS <- -std=c++11")
else ()
    #    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++14")
endif ()
################################
# set lib version here

set(GENERIC_LIB_VERSION "5.0.1")
set(GENERIC_LIB_SOVERSION "1")

####################################
set(MATH21_WORKING_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../zmath21)
set(MATH21_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/includes)
####################################
set(BUILD_SHARED_LIBS ON)

set(MATH21_FLAG_USE_THREAD ON)
#set(MATH21_FLAG_USE_OPENMP ON)
set(LI_INSTALL_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/z)
set(CMAKE_INSTALL_PREFIX ${LI_INSTALL_LOCATION})

set(MATH21_CUDA_HOME /usr/local/cuda)
#set(MATH21_CUDA_HOME /usr/local/cuda-10.0)

# tmp
#set(MATH21_FLAG_USE_OPENCL_BLAS ON)

# for cuda
set(ENV{CUDACXX} ${MATH21_CUDA_HOME}/bin/nvcc)
# for opencl
set(ENV{CUDA_PATH} ${MATH21_CUDA_HOME})

#set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################ system ################
if (WIN32)
    set(MATH21_FLAG_IS_WIN32 ON)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(MATH21_FLAG_IS_WIN_MSVC ON)
    endif ()
elseif (ANDROID)
    set(MATH21_FLAG_IS_ANDROID ON)
elseif (APPLE)
    set(MATH21_FLAG_IS_APPLE ON)
elseif (UNIX)
    set(MATH21_FLAG_IS_LINUX ON)
else ()
    message("operating systems unknown")
    set(MATH21_FLAG_IS_LINUX ON)
endif ()

########## disable #############

if (DISABLE_ANDROID) # set by -DDISABLE_ANDROID
    if (MATH21_FLAG_IS_ANDROID)
        message("Defining Linux instead of Android")
        set(MATH21_FLAG_IS_ANDROID OFF)
        set(MATH21_FLAG_IS_LINUX ON)
    endif ()
endif ()

########### message #############
if (MATH21_FLAG_IS_WIN32)
    message("IS_WIN32 ON")
    if (MATH21_FLAG_IS_WIN_MSVC)
        message("IS_WIN_MSVC ON")
    endif ()
elseif (MATH21_FLAG_IS_ANDROID)
    message("IS_ANDROID ON")
elseif (MATH21_FLAG_IS_APPLE)
    message("IS_APPLE ON")
elseif (MATH21_FLAG_IS_LINUX)
    message("IS_LINUX ON")
endif ()
################################
if (MATH21_FLAG_IS_APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
endif ()

if (MATH21_FLAG_IS_ANDROID)
    find_library( # Sets the name of the path variable.
            log-lib
            # Specifies the name of the NDK library that
            # you want CMake to locate.
            log)
endif ()

set(MATH21_DEPENDENT_LIBS ${MATH21_DEPENDENT_LIBS} ${log-lib})

if (MATH21_FLAG_IS_LINUX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif ()
################################
# Add definitions
if (1)
    if (MATH21_FLAG_EXTERNAL)
        option(MATH21_FLAG_RELEASE "Enable Release" ${MATH21_FLAG_EXTERNAL_RELEASE})
        option(MATH21_FLAG_USE_CUDA "Use CUDA" ${MATH21_FLAG_EXTERNAL_USE_CUDA})
        option(MATH21_FLAG_USE_OPENCL "Use OpenCL" ${MATH21_FLAG_EXTERNAL_USE_OPENCL})
    else ()
        option(MATH21_FLAG_RELEASE "Enable Release" ON)
        option(MATH21_FLAG_USE_CUDA "Use CUDA" ON)
        option(MATH21_FLAG_USE_OPENCL "Use OpenCL" ON)
    endif ()

    if (MATH21_FLAG_IS_WIN32)
    else ()
        #option(MATH21_FLAG_USE_OPENMP "OPENMP" ON)
    endif ()

    # test openmp
    if (MATH21_FLAG_USE_CUDA)
        #        option(MATH21_FLAG_USE_OPENMP "OPENMP" ON)
    endif ()
endif ()

################ set ################
if (${BUILD_SHARED_LIBS})
    set(MATH21_LIB_SHARED ON)
else ()
    set(MATH21_LIB_SHARED OFF)
endif ()

if (MATH21_FLAG_IS_WIN_MSVC)
    if (${MATH21_LIB_SHARED})
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    endif ()
endif ()
################ message ################
if (${MATH21_FLAG_EXTERNAL})
    message(STATUS "MATH21_FLAG_EXTERNAL: ${MATH21_FLAG_EXTERNAL}")
else ()
    message(STATUS "MATH21_FLAG_EXTERNAL: OFF")
endif ()
if (${MATH21_FLAG_USE_CUDA})
    message(STATUS "MATH21_FLAG_USE_CUDA: ${MATH21_FLAG_USE_CUDA}")
else ()
    message(STATUS "MATH21_FLAG_USE_CUDA: OFF")
endif ()
if (${MATH21_FLAG_USE_OPENCL})
    message(STATUS "MATH21_FLAG_USE_OPENCL: ${MATH21_FLAG_USE_OPENCL}")
else ()
    message(STATUS "MATH21_FLAG_USE_OPENCL: OFF")
endif ()
if (${MATH21_FLAG_USE_OPENMP})
    message(STATUS "MATH21_FLAG_USE_OPENMP: ${MATH21_FLAG_USE_OPENMP}")
else ()
    message(STATUS "MATH21_FLAG_USE_OPENMP: OFF")
endif ()
if (${MATH21_FLAG_USE_THREAD})
    message(STATUS "MATH21_FLAG_USE_THREAD: ${MATH21_FLAG_USE_THREAD}")
else ()
    message(STATUS "MATH21_FLAG_USE_THREAD: OFF")
endif ()
message(STATUS "MATH21_LIB_SHARED: ${MATH21_LIB_SHARED}")
message(STATUS "MATH21_LIB_VERSION: ${GENERIC_LIB_VERSION}")
################ check ################
if (MATH21_FLAG_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        #            message(FATAL_ERROR "CHECK")
        enable_language(CUDA)
        message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")
        if (NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_STANDARD 11)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        endif ()
    else ()
        #        message(FATAL_ERROR "CHECK2")
        set(MATH21_FLAG_USE_CUDA OFF)
        message(STATUS "CUDA not found, MATH21_FLAG_USE_CUDA: OFF")
    endif ()
endif ()

if (MATH21_FLAG_USE_CUDA)
    if (MATH21_FLAG_USE_OPENCL)
        set(MATH21_FLAG_USE_OPENCL OFF)
        message(STATUS "OpenCL is disabled because CUDA is enabled.")
    endif ()
endif ()

if (MATH21_FLAG_USE_OPENCL)
    #    ENV INTELOCLSDKROOT, CUDA_PATH
    find_package(OpenCL)
    message(STATUS "math21 OpenCL found: ${OpenCL_FOUND}")
    if (${OpenCL_FOUND})
        message(STATUS "math21 OpenCL_INCLUDE_DIRS: ${OpenCL_INCLUDE_DIRS}")
        message(STATUS "math21 OpenCL_LIBRARIES: ${OpenCL_LIBRARIES}")
        message(STATUS "math21 OpenCL_VERSION_STRING: ${OpenCL_VERSION_STRING}")
    else ()
        set(MATH21_FLAG_USE_OPENCL OFF)
    endif ()
endif ()

################ set lib ################
if (${MATH21_FLAG_USE_CUDA})
    set(MATH21_DEPENDENT_INCLUDE_CUDA ${MATH21_CUDA_HOME}/include)
    set(MATH21_DEPENDENT_LIB_CUDA -L${MATH21_CUDA_HOME}/lib64 -lcuda -lcudart -lcublas -lcurand)
    #    set(MATH21_GLOBAL_INCLUDE_01 ${MATH21_DEPENDENT_INCLUDE_CUDA} PARENT_SCOPE)
endif ()

if (MATH21_FLAG_USE_OPENCL)
    # tmp
    if (MATH21_FLAG_USE_OPENCL_BLAS)
        set(MATH21_DEPENDENT_INCLUDE_OPENCL_BLAS ../clBLAS)
        if (MATH21_FLAG_IS_LINUX)
            link_libraries(/mathxyz/libclBLAS.so)
        else ()
            set(MATH21_FLAG_USE_OPENCL_BLAS OFF)
        endif ()
    endif ()
    set(MATH21_DEPENDENT_INCLUDE_OPENCL ${OpenCL_INCLUDE_DIRS})
    set(MATH21_DEPENDENT_LIB_OPENCL ${OpenCL_LIBRARIES})
else ()
    set(MATH21_FLAG_USE_OPENCL_BLAS OFF)
endif ()

set(MATH21_DEPENDENT_INCLUDES ${MATH21_DEPENDENT_INCLUDES}
        ${MATH21_DEPENDENT_INCLUDE_CUDA} ${MATH21_DEPENDENT_INCLUDE_OPENCL} ${MATH21_DEPENDENT_INCLUDE_OPENCL_BLAS})

set(MATH21_DEPENDENT_LIBS ${MATH21_DEPENDENT_LIBS} ${MATH21_DEPENDENT_LIB_CUDA} ${MATH21_DEPENDENT_LIB_OPENCL})

if (NOT ${MATH21_FLAG_IS_WIN_MSVC})
    set(MATH21_DEPENDENT_LIBS ${MATH21_DEPENDENT_LIBS} -lm)
endif ()

if (${MATH21_FLAG_USE_THREAD})
    set(MATH21_DEPENDENT_LIBS ${MATH21_DEPENDENT_LIBS} -pthread)
endif ()

################################
if (${MATH21_FLAG_USE_OPENMP})
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        if (MATH21_FLAG_IS_WIN_MSVC)
            set(MATH21_FLAG_USE_OPENMP OFF)
            message(STATUS "OpenMP is disabled because of low version. MATH21_FLAG_USE_OPENMP: OFF")
        endif ()
    else ()
        set(MATH21_FLAG_USE_OPENMP OFF)
        message(STATUS "OpenMP not found, MATH21_FLAG_USE_OPENMP: OFF")
    endif ()
endif ()
################################
#if (MATH21_FLAG_USE_OPENMP)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#endif ()
################ configure file ################
if (${MATH21_FLAG_EXTERNAL})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/math21_user_config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/math21_external_user_config_generated @ONLY)
else ()
    add_definitions("-DMATH21_FLAG_NOT_EXTERNAL")
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/math21_user_config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/math21_user_config_generated @ONLY)
endif ()
################################

include_directories(..)
include_directories(includes)
set(module_name math21)

if (MATH21_FLAG_USE_CUDA)
    FILE(GLOB_RECURSE sourcefiles "3rdparty/*.c" "3rdparty/*.cc" "3rdparty/*.cu" "src/*.c" "src/*.cc" "src/*.cu")
else ()
    FILE(GLOB_RECURSE sourcefiles "3rdparty/*.c" "3rdparty/*.cc" "src/*.c" "src/*.cc")
endif ()

if (MATH21_FLAG_IS_ANDROID)
    add_library(${module_name} SHARED ${sourcefiles})
else ()
    add_library(${module_name} ${sourcefiles})
endif ()

if (MATH21_FLAG_IS_WIN32)
    if (NOT MATH21_FLAG_IS_WIN_MSVC)
        ## cross compile only
        target_compile_options(${module_name} PRIVATE -fno-exceptions)
    endif ()
endif ()

if (${MATH21_FLAG_USE_CUDA})
    set(CMAKE_CUDA_ARCHITECTURES OFF)
    set_property(TARGET ${module_name} PROPERTY CUDA_ARCHITECTURES OFF)
    target_compile_options(${module_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--relocatable-device-code=true>)
    set_property(TARGET ${module_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif ()

if (${MATH21_FLAG_USE_OPENMP})
    #    target_link_libraries(${module_name} PUBLIC OpenMP::OpenMP_CXX ${MATH21_DEPENDENT_LIBS})
    target_link_libraries(${module_name} OpenMP::OpenMP_CXX ${MATH21_DEPENDENT_LIBS})
else ()
    target_link_libraries(${module_name} ${MATH21_DEPENDENT_LIBS})
endif ()

target_include_directories(${module_name} SYSTEM PUBLIC ${MATH21_DEPENDENT_INCLUDES})

#set_target_properties(${module_name} PROPERTIES
#        COMPILE_DEFINITIONS "MATH21_EXPORT"
#        VERSION "${GENERIC_LIB_VERSION}"
#        SOVERSION "${GENERIC_LIB_SOVERSION}")

################ install ################

install(TARGETS ${module_name}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)

################ test ################
set(module_test_name math21_test)

if (MATH21_FLAG_IS_ANDROID)
else ()
    if (MATH21_FLAG_USE_CUDA)
        FILE(GLOB_RECURSE SOURCE_FILES "test/*.c" "test/*.cc" "test/*.cu")
    else ()
        FILE(GLOB_RECURSE SOURCE_FILES "test/*.c" "test/*.cc")
    endif ()

    add_executable(${module_test_name} ${SOURCE_FILES})
    if (MATH21_FLAG_IS_WIN32)
        if (MATH21_FLAG_IS_WIN_MSVC)
            target_link_libraries(${module_test_name} ${module_name})
        else ()
            ## cross compile only
            target_link_libraries(${module_test_name} ${module_name} -static)
            target_compile_options(${module_test_name} PRIVATE -fno-exceptions)
        endif ()
    else ()
        target_link_libraries(${module_test_name} ${module_name})
    endif ()
endif ()
###### install #######
if (MATH21_FLAG_IS_ANDROID)
else ()
    install(TARGETS ${module_test_name}
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib/static)
endif ()
################ ctest ################
#enable_testing()
#add_subdirectory(src/numbers)
add_subdirectory(unit_test/numbers)
################ MATH21_AS_LIB ################
# windows failed
set(MATH21_AS_LIB math21_imported)
# only shared libs importing is supported when gpu.
add_library(${MATH21_AS_LIB} SHARED IMPORTED GLOBAL)
set_target_properties(${MATH21_AS_LIB} PROPERTIES IMPORTED_LOCATION /mathxyz/libmath21.so)
if (MATH21_FLAG_IS_ANDROID)
    # android failed
elseif (MATH21_FLAG_IS_LINUX)
    target_include_directories(${MATH21_AS_LIB} SYSTEM INTERFACE ${MATH21_DEPENDENT_INCLUDES})
endif ()