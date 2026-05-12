set(TARGET llama-cli)

# spiral_metal_capture.mm is Mac-only Objective-C++ (uses MTLCaptureManager).
# Its lone caller in cli.cpp is already wrapped in #ifdef __APPLE__, so on
# non-Apple platforms we simply don't compile or link the .mm file or its
# Apple frameworks. cli.cpp builds and runs identically on both platforms;
# GPU capture is just unavailable off Mac (and not meaningful there anyway —
# CUDA capture uses Nsight Systems, not MTLCaptureManager).
if(APPLE)
    add_executable(${TARGET} cli.cpp spiral_metal_capture.mm)
    target_link_libraries(${TARGET} PRIVATE "-framework Metal" "-framework Foundation")
else()
    add_executable(${TARGET} cli.cpp)
endif()

target_link_libraries(${TARGET} PRIVATE server-context PUBLIC common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

include_directories(../server)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()