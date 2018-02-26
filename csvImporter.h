/******************************************************************************
* Copyright (c) 2016-2018, Brian Kennedy.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
******************************************************************************/

/******************************************************************************
*
* See https://github.com/Simantex/CSVImporter for repository and documentation.
*
******************************************************************************/
// System includes

#ifndef CSVIMPORTER_H
#define CSVIMPORTER_H

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <time.h>
#ifdef WIN32
#include <io.h>
#include <windows.h>
#endif
#ifndef WIN32
#include <sys/mman.h> // for mmap() / munmap()
#endif
using namespace std;

// CUDA runtime
#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

// CUDA atomics
#include "sm_35_atomic_functions.h"

// CUDA helper functions and utilities to work with CUDA
#include "helper_cuda.h"

#include "CommonDefinitions.h"

#define DEFAULT_PINNED_GENERIC_MEMORY true

// Macro to aligned up to the memory size in question
#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )

extern bool bPinGenericMemory;
extern uint32_t device_sync_method;

// CPU Helper routines to ensure we allocate memory on the CPU that the GPUs are associated with
void AllocateHostMemory(bool bPinGenericMemory, uint32_t **pp_a, uint32_t **ppAligned_a, unsigned int nbytes, bool bUseWriteCombine);
void FreeHostMemory(bool bPinGenericMemory, uint32_t **pp_a, uint32_t **ppAligned_a, unsigned int nbytes);

extern "C" bool InitializeCPUElements_REUSABLES(uint64_t totalbytes);
extern "C" void DeinitializeCPUElements_REUSABLES(uint64_t totalbytes);

extern "C" int CSVfilechunking(char * filepath);

extern "C" bool InitializeGPUElements_REUSABLES(uint64_t totalbytes);
extern "C" bool MemsetGPUElements_REUSABLES(uint64_t totalbytes);
extern "C" void DeinitializeGPUElements_REUSABLES();


extern uint8_t *h_CsvBuffer_NAXX;
extern uint8_t *h_CsvBuffer_a; // aligned

extern uint32_t *h_RecordHeaders_NAXX;
extern uint32_t *h_RecordHeaders_a; // aligned
extern uint32_t *h_ColumnHeaders_NAXX;
extern uint32_t *h_ColumnHeaders_a; // aligned
extern uint32_t *h_RecordHeadersSCAN_NAXX;
extern uint32_t *h_RecordHeadersSCAN_a; // aligned
extern uint32_t *h_ColumnHeadersSCAN_NAXX;
extern uint32_t *h_ColumnHeadersSCAN_a; // aligned
extern uint32_t *h_RecordsTable_NAXX;
extern uint32_t *h_RecordsTable_a; // aligned
extern uint32_t *h_RecordsToColumnsTable_NAXX;
extern uint32_t *h_RecordsToColumnsTable_a; // aligned
extern uint32_t *h_ColumnsTable_NAXX;
extern uint32_t *h_ColumnsTable_a; // aligned
extern uint32_t *h_RecordLengths_NAXX;
extern uint32_t *h_RecordLengths_a; // aligned
extern uint32_t *h_ColumnCountErrors_NAXX;
extern uint32_t *h_ColumnCountErrors_a; // aligned

extern uint32_t *h_ColumnCharCountErrors_NAXX;
extern uint32_t *h_ColumnCharCountErrors_a; // aligned

extern uint32_t *h_ColumnCountErrorsSCAN_NAXX;
extern uint32_t *h_ColumnCountErrorsSCAN_a; // aligned
extern uint32_t *h_ColumnCountErrorsTable_NAXX;
extern uint32_t *h_ColumnCountErrorsTable_a; // aligned
extern uint32_t *h_ColumnCountsPerRecordTable_NAXX;
extern uint32_t *h_ColumnCountsPerRecordTable_a; // aligned

extern uint64_t CsvFileLength;
extern uint64_t SufficientBytes;

//extern uint32_t *d_CsvBuffer;
extern uint8_t *d_CsvBuffer;

extern uint32_t *d_RecordHeaders;
extern uint32_t *d_ColumnHeaders;
extern uint32_t *d_RecordHeadersSCAN;
extern uint32_t *d_ColumnHeadersSCAN;

extern uint32_t *d_RecordsTable;
extern uint32_t *d_RecordsToColumnsTable;
extern uint32_t *d_ColumnsTable;
extern uint32_t *d_RecordLengths;
extern uint32_t *d_ColumnCountErrors;
extern uint32_t *d_ColumnCharCountErrors;

extern uint32_t *d_ColumnCountErrorsSCAN;
extern uint32_t *d_ColumnCountErrorsTable;
extern uint32_t *d_ColumnCountsPerRecordTable;

extern uint32_t *d_QuoteBoundaryHeaders;  // flags for column boundaries with quotes.
extern uint32_t *d_QuoteBoundaryHeaders_SCAN;  // flags for column boundaries with quotes.
extern uint32_t *d_printingchars_flags;  // flags for printing chars (first flag non-printing, then reverse the meaning).
extern uint32_t *d_printingchars_SCAN;  // flags for printing chars (first flag non-printing, then reverse the meaning).
extern uint8_t *d_CommaHeaders;  // single byte flag for comma headers (comma found, possible col boundary).
extern uint8_t *d_secondquotes;  // single byte flag for 2nd of "" -- used to flag non-printing.
extern uint32_t * d_RecordsToQuoteBoundariesTable;

extern uint32_t *h_UTF8Headers_NAXX;
extern uint32_t *h_UTF8Headers_a;  // aligned
extern uint32_t *h_UTF8HeadersSCAN_NAXX;
extern uint32_t *h_UTF8HeadersSCAN_a;  // aligned
extern uint32_t *h_UTF8CharsTable_NAXX;
extern uint32_t *h_UTF8CharsTable_a;  // aligned
extern uint32_t *h_ColumnsToUTF8charsTable_NAXX;
extern uint32_t *h_ColumnsToUTF8charsTable_a;  // aligned
extern uint16_t *h_CharNumInColumn_NAXX;
extern uint16_t *h_CharNumInColumn_a;  // aligned

extern uint16_t *h_ColumnNumInRecord_NAXX;
extern uint16_t *h_ColumnNumInRecord_a; // aligned

extern uint32_t *d_UTF8Headers;
extern uint32_t *d_UTF8HeadersSCAN;
extern uint32_t *d_UTF8CharsTable;
extern uint32_t *d_ColumnsToUTF8charsTable;
extern uint16_t *d_ColumnNumInRecord;
extern uint16_t *d_CharNumInColumn;

extern uint8_t *  d_CsvBuffer_printing;  // shortened CSV buffer.
extern uint32_t *  d_RecordHeaders_printing;  // record headers shortened.
extern uint32_t *  d_ColumnHeaders_printing;  // col headers shortened.

extern cudaError_t cudaStatus;

#define MAXCHUNKS 100
#define TARGETBYTES 50000000
#define OVERREAD 1000192
#define maxsortvals 12

extern int inumchunks;
extern uint64_t apprx_chunklen;

extern FILE * pCsvFileIn;
extern int64_t startseek;  // place to start in file to read a chunk

extern uint32_t chunkrecidxstarts[MAXCHUNKS + 1];
extern uint32_t chunkbufidxstarts[MAXCHUNKS + 1];
extern uint32_t curchunkreccounts[MAXCHUNKS];
extern uint32_t curchunkbufbytecounts[MAXCHUNKS];

extern uint32_t maxchunkreccount;
extern uint32_t maxchunkbufbytecount;

extern uint32_t totalrecordscount;

struct memlinklist_member
{
	memlinklist_member * next;
	void * h_cur_block_ptr;
	int cur_block_validrecordcount;  // this is count of records WITHOUT errors in block
	int cur_block_errorcount;  // combined col count and char count error count.
};

struct memlinklist_base
{
	memlinklist_member * first;
	int recordwidth;
	int totalvalidrecords;  // this is total count of records WITHOUT errors
	int totalerrors;  // total combined col count and char count error count.
};

extern memlinklist_base * h_llbases;

#define MAXNUMCOLUMNS 255

extern const char *fieldnames[MAXNUMCOLUMNS];
extern int16_t * h_fieldUTF8charsizes;
extern uint16_t G_h_fieldbytewidths[MAXNUMCOLUMNS];

extern unsigned char ** h_fieldptrs;
extern unsigned char ** h_d_fieldptrs;

extern int nullterm_addchar;  // add a char to width for a null term (optional).

// count of total valid CSV records.
extern uint64_t G_totalvalidCSVrecordscount;

// global for indices (originally a retained sort index, later simple array indices used so index always == ix).
extern uint64_t * G_d_indices;

extern char * G_h_d_strings_start_ptr_C[maxsortvals];
extern uint64_t G_h_d_bytes_per_string_C[maxsortvals];
extern int G_h_d_sort_defaultup_C[maxsortvals];
extern bool G_h_d_sortspacetoend_C[maxsortvals];
extern char G_h_d_sort_datatype_C[maxsortvals];  // the type.
extern char h_d_values_count_C[1];  // this is just the count of values.  (max is maxsortvals per array sizes above.)

extern clock_t startTime;
extern clock_t testTime;
extern clock_t timePassed;
extern double secondsPassed;

extern int validrecordscount;

extern "C" void Check_cuda_Free(void ** memlocaddress);
extern "C" void Check_cuda_FreeHost(void ** memlocaddress);

#define Check_cuda_Errors(message)      __Check_cuda_Errors (message, __FILE__, __LINE__)
extern "C" void __Check_cuda_Errors(const char *errorMessage, const char *file, const int line);

#endif
