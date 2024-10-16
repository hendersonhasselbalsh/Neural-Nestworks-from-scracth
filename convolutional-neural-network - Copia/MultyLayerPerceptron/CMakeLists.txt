set(PROJECT_NAME MultyLayerPerceptron)

################################################################################
# Source groups
################################################################################
set(no_group_source_files
    "main.cpp"
)
source_group("" FILES ${no_group_source_files})

set(cnn__Headers
    "cnn/ActivationCell.h"
    "cnn/cnn.h"
    "cnn/CNNbuilder.h"
    "cnn/ConvolutionCell.h"
    "cnn/data-maneging.h"
    "cnn/layer-normalization.h"
    "cnn/pooling.h"
    "cnn/ProcessingUnity.h"
)
source_group("cnn\\Headers" FILES ${cnn__Headers})

set(cnn__Sources
    "cnn/ActivationCell.cpp"
    "cnn/cnn.cpp"
    "cnn/CNNbuilder.cpp"
    "cnn/ConvolutionCell.cpp"
    "cnn/data-maneging.cpp"
    "cnn/layer-normalization.cpp"
    "cnn/pooling.cpp"
)
source_group("cnn\\Sources" FILES ${cnn__Sources})

set(mlp__Headers
    "mlp/activation-functions.h"
    "mlp/layer.h"
    "mlp/lost-function.h"
    "mlp/mlp-builder.h"
    "mlp/multy-layer-perceptron.h"
)
source_group("mlp\\Headers" FILES ${mlp__Headers})

set(mlp__Sources
    "mlp/activation-functions.cpp"
    "mlp/layer.cpp"
    "mlp/lost-function.cpp"
    "mlp/mlp-builder.cpp"
    "mlp/multy-layer-perceptron.cpp"
)
source_group("mlp\\Sources" FILES ${mlp__Sources})

set(utils__Headers
    "utils/basic-includes.h"
    "utils/utils.h"
)
source_group("utils\\Headers" FILES ${utils__Headers})

set(utils__Sources
    "utils/utils.cpp"
)
source_group("utils\\Sources" FILES ${utils__Sources})

set(ALL_FILES
    ${no_group_source_files}
    ${cnn__Headers}
    ${cnn__Sources}
    ${mlp__Headers}
    ${mlp__Sources}
    ${utils__Headers}
    ${utils__Sources}
)

################################################################################
# Target
################################################################################
add_executable(${PROJECT_NAME} ${ALL_FILES})

use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
set(ROOT_NAMESPACE MultyLayerPerceptron)

set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_GLOBAL_KEYWORD "Win32Proj"
)
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
    )
endif()
################################################################################
# Include directories
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.dependency/my-custom-made-gnuplot;"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.dependency/opencv-4.8.1/include;"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.dependency/json-develop/include;"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.dependency/eigen-3.4.0"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
    target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.dependency/my-custom-made-gnuplot;"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.dependency/opencv-4.8.1/include;"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.dependency/json-develop/include;"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.dependency/eigen-3.4.0"
    )
endif()

################################################################################
# Compile definitions
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "_DEBUG"
        ">"
        "$<$<CONFIG:Release>:"
            "NDEBUG"
        ">"
        "_CONSOLE;"
        "UNICODE;"
        "_UNICODE"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "_DEBUG"
        ">"
        "$<$<CONFIG:Release>:"
            "NDEBUG"
        ">"
        "WIN32;"
        "_CONSOLE;"
        "UNICODE;"
        "_UNICODE"
    )
endif()

################################################################################
# Compile and link options
################################################################################
if(MSVC)
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /W3
            >
            $<$<CONFIG:Release>:
                /Oi;
                /Gy;
                /w
            >
            /permissive-;
            /sdl;
            ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
            ${DEFAULT_CXX_EXCEPTION_HANDLING}
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Release>:
                /Oi;
                /Gy
            >
            /permissive-;
            /sdl;
            /W3;
            ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
            ${DEFAULT_CXX_EXCEPTION_HANDLING}
        )
    endif()
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_link_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Release>:
                /OPT:REF;
                /OPT:ICF
            >
            /DEBUG;
            /SUBSYSTEM:CONSOLE
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
        target_link_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Release>:
                /OPT:REF;
                /OPT:ICF
            >
            /DEBUG;
            /SUBSYSTEM:CONSOLE
        )
    endif()
endif()

################################################################################
# Pre build events
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    add_custom_command_if(
        TARGET ${PROJECT_NAME}
        PRE_BUILD
        COMMANDS
        COMMAND   $<CONFIG:Debug> copy "..\\..\\.dependency\\opencv-4.8.1\\bin\\opencv_world480d.dll" "$<SHELL_PATH:${OUTPUT_DIRECTORY}>"
        COMMAND $<CONFIG:Release> copy "..\\..\\.dependency\\opencv-4.8.1\\bin\\opencv_world480.dll" "$<SHELL_PATH:${OUTPUT_DIRECTORY}>"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
    add_custom_command_if(
        TARGET ${PROJECT_NAME}
        PRE_BUILD
        COMMANDS
        COMMAND   $<CONFIG:Debug> copy "..\\..\\.dependency\\opencv-4.8.1\\bin\\opencv_world480d.dll" "$<SHELL_PATH:${OUTPUT_DIRECTORY}>"
        COMMAND $<CONFIG:Release> copy "..\\..\\.dependency\\opencv-4.8.1\\bin\\opencv_world480.dll" "$<SHELL_PATH:${OUTPUT_DIRECTORY}>"
    )
endif()

################################################################################
# Dependencies
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    set(ADDITIONAL_LIBRARY_DEPENDENCIES
        "$<$<CONFIG:Debug>:"
            "opencv_world480d"
        ">"
        "$<$<CONFIG:Release>:"
            "opencv_world480"
        ">"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
    set(ADDITIONAL_LIBRARY_DEPENDENCIES
        "$<$<CONFIG:Debug>:"
            "opencv_world480d"
        ">"
        "$<$<CONFIG:Release>:"
            "opencv_world480"
        ">"
    )
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE "${ADDITIONAL_LIBRARY_DEPENDENCIES}")

if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_link_directories(${PROJECT_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.dependency/opencv-4.8.1/lib"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
    target_link_directories(${PROJECT_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.dependency/opencv-4.8.1/lib"
    )
endif()

