set (module_name math21_opt)
message(STATUS "${module_name}")

#FILE(GLOB sourcefiles "*.cc" "*.c")
FILE(GLOB_RECURSE sourcefiles "*.cc" "*.c")

#add_library(${module_name} SHARED ${sourcefiles})
add_library(${module_name} STATIC ${sourcefiles})

target_link_libraries(${module_name} math21_ad)

install (TARGETS ${module_name}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)

