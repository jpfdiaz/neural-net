if(NOT ENABLE_COVERAGE)
    return()
endif()

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(WARNING "Coverage requires GCC or Clang — skipping")
    return()
endif()

add_compile_options(--coverage)
add_link_options(--coverage)
message(STATUS "Coverage instrumentation enabled")

find_program(LCOV_BIN    lcov)
find_program(GENHTML_BIN genhtml)

if(LCOV_BIN AND GENHTML_BIN)
    set(COVERAGE_RAW    ${CMAKE_BINARY_DIR}/coverage_raw.info)
    set(COVERAGE_INFO   ${CMAKE_BINARY_DIR}/coverage.info)
    set(COVERAGE_REPORT ${CMAKE_SOURCE_DIR}/coverage)

    add_custom_target(coverage
        # Capture all instrumented data from the build tree
        COMMAND ${LCOV_BIN}
                --capture
                --directory ${CMAKE_BINARY_DIR}
                --output-file ${COVERAGE_RAW}
                --gcov-tool gcov
                --ignore-errors inconsistent
        # Keep only project sources under src/; suppress warnings for
        # files that exist on disk but weren't compiled (e.g. CMakeLists.txt)
        COMMAND ${LCOV_BIN}
                --extract ${COVERAGE_RAW}
                "${CMAKE_SOURCE_DIR}/src/*"
                --output-file ${COVERAGE_INFO}
                --ignore-errors unused,unused
        COMMAND ${GENHTML_BIN}
                ${COVERAGE_INFO}
                --output-directory ${COVERAGE_REPORT}
        COMMAND ${CMAKE_COMMAND} -E echo
                "Coverage report: ${COVERAGE_REPORT}/index.html"
        COMMENT "Generating HTML coverage report"
    )
else()
    message(STATUS "lcov/genhtml not found — install lcov to enable the 'coverage' target")
endif()