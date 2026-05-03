set(TARGET llama-cli)
add_executable(${TARGET} cli.cpp spiral_metal_capture.mm)
target_link_libraries(${TARGET} PRIVATE "-framework Metal" "-framework Foundation")

target_link_libraries(${TARGET} PRIVATE server-context PUBLIC common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

include_directories(../server)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
