set (module_name math21)
message(STATUS "${module_name}")


include_directories(../includes)

add_subdirectory(functions)
add_subdirectory(numbers)
add_subdirectory(print)
add_subdirectory(memory)
add_subdirectory(matrix)
add_subdirectory(matrix_analysis)
add_subdirectory(matrix_op)
add_subdirectory(algebra)
add_subdirectory(variables)
add_subdirectory(ad)
add_subdirectory(opt)

FILE(GLOB sourcefiles "*.cc" "*.c")
#FILE(GLOB_RECURSE sourcefiles "*.cc" "*.c")

#add_library(${module_name} SHARED ${sourcefiles})
add_library(${module_name} STATIC ${sourcefiles})

#target_link_libraries(${module_name}
#        math21_opt
#        math21_ad
#        math21_functions
#        math21_variables
#        math21_algebra
#        math21_matrix_op
#        math21_matrix_analysis
#        math21_matrix
#        math21_print
#        math21_numbers
#        )

install (TARGETS ${module_name}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)

