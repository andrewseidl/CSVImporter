cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CSVImporter LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)

include_directories($CMAKE_SOURCE_DIR ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_library(csv_importer STATIC
  BuildRecsColsCharsHeaders.cu
  CommaProcessing.cu
  GetCharNumsInColumns.cu
  GetColumnNumsInRecords.cu
  GetRecLengthsAndColCountErrors.cu
  MainLauncher.cu
  RecordsColumnsChars_StreamCompact.cu
  SimpleStreamCompact.cu
  WriteCSVRecord2.cu
  csvCPUfuncs.cpp
  csvGlobals.cpp
  csvGPUfuncs.cpp
  src/mgpucontext.cu
  src/mgpuutil.cpp
  src/mgpucontext.cu
  )

add_executable(main Main.cpp MainLauncher.cu)
target_link_libraries(main csv_importer)
