set(module_name math21_numbers_test)
message(STATUS "${module_name}")


#add_subdirectory(ad)
#add_subdirectory(opt)
#add_subdirectory(tensor)

#FILE(GLOB sourcefiles "*.cc" "*.c")
FILE(GLOB_RECURSE sourcefiles "*.cc" "*.c")

add_executable(${module_name} ${sourcefiles})

#target_link_libraries(${module_name}
#        math21_numbers)

target_link_libraries(${module_name}
        math21)

add_test(NAME ${module_name} COMMAND ${module_name})

#install(TARGETS ${module_name}
#        RUNTIME DESTINATION bin
#        LIBRARY DESTINATION lib
#        ARCHIVE DESTINATION lib/static)
