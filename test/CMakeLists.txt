include_directories(../includes)
set (module_name math21_test)
message(STATUS "${module_name}")


#add_subdirectory(ad)
#add_subdirectory(opt)
#add_subdirectory(tensor)

#FILE(GLOB sourcefiles "*.cc" "*.c")
FILE(GLOB_RECURSE sourcefiles "*.cc" "*.c")

add_executable(${module_name} ${sourcefiles})
target_link_libraries(${module_name}
#        m21
        math21_opt
        math21_ad
        math21_functions
        math21_variables
        math21_algebra
        math21_matrix_op
        math21_matrix_analysis
        math21_matrix
        math21_print
        math21_numbers

#        math21_ad_test
#        math21_opt_test
#        math21_tensor_test
        )

install (TARGETS ${module_name}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)
