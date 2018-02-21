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
#include "csvImporter.h"


bool bPinGenericMemory = DEFAULT_PINNED_GENERIC_MEMORY; // we want this to be the default behavior


/* Valid settings for device_sync method:
0 cudaDeviceScheduleAuto  (Automatic Blocking)
1 cudaDeviceScheduleSpin  (Spin Blocking)
2 cudaDeviceScheduleYield  (Yield Blocking)
3 (Undefined Blocking Method  DO NOT USE)
4 cudaDeviceBlockingSync  (Blocking Sync Event) = low CPU utilization
*/
uint32_t  device_sync_method = cudaDeviceBlockingSync; // by default we use BlockingSync

uint8_t *h_CsvBuffer_NAXX = 0;
uint8_t *h_CsvBuffer_a = 0; // aligned

uint32_t *h_RecordHeaders_NAXX = 0;
uint32_t *h_RecordHeaders_a = 0; // aligned
uint32_t *h_ColumnHeaders_NAXX = 0;
uint32_t *h_ColumnHeaders_a = 0; // aligned

uint32_t *h_RecordsTable_NAXX = 0;
uint32_t *h_RecordsTable_a = 0; // aligned
uint32_t *h_RecordsToColumnsTable_NAXX = 0;
uint32_t *h_RecordsToColumnsTable_a = 0; // aligned
uint32_t *h_ColumnsTable_NAXX = 0;
uint32_t *h_ColumnsTable_a = 0; // aligned
uint32_t *h_RecordLengths_NAXX = 0;
uint32_t *h_RecordLengths_a = 0; // aligned
uint32_t *h_ColumnCountErrors_NAXX = 0;
uint32_t *h_ColumnCountErrors_a = 0; // aligned
uint32_t *h_ColumnCharCountErrors_NAXX = 0;
uint32_t *h_ColumnCharCountErrors_a = 0; // aligned

uint32_t *h_ColumnCountErrorsSCAN_NAXX = 0;
uint32_t *h_ColumnCountErrorsSCAN_a = 0; // aligned
uint32_t *h_ColumnCountErrorsTable_NAXX = 0;
uint32_t *h_ColumnCountErrorsTable_a = 0; // aligned
uint32_t *h_ColumnCountsPerRecordTable_NAXX = 0;
uint32_t *h_ColumnCountsPerRecordTable_a = 0; // aligned


uint32_t *h_RecordHeadersSCAN_NAXX = 0;
uint32_t *h_RecordHeadersSCAN_a = 0; // aligned
uint32_t *h_ColumnHeadersSCAN_NAXX = 0;
uint32_t *h_ColumnHeadersSCAN_a = 0; // aligned

uint64_t CsvFileLength = 0;
uint64_t SufficientBytes = 0;

//uint32_t *d_CsvBuffer = 0;
uint8_t *d_CsvBuffer = 0;

uint32_t *d_RecordHeaders = 0;
uint32_t *d_ColumnHeaders = 0;
uint32_t *d_RecordHeadersSCAN = 0;
uint32_t *d_ColumnHeadersSCAN = 0;
uint32_t *d_RecordsTable = 0;
uint32_t *d_RecordsToColumnsTable = 0;
uint32_t *d_ColumnsTable = 0;
uint32_t *d_RecordLengths = 0;
uint32_t *d_ColumnCountErrors = 0;
uint32_t *d_ColumnCharCountErrors = 0;

uint32_t *d_ColumnCountErrorsSCAN = 0;
uint32_t *d_ColumnCountErrorsTable = 0;
uint32_t *d_ColumnCountsPerRecordTable = 0;

uint32_t *d_QuoteBoundaryHeaders = 0;  // flags for column boundaries with quotes.
uint32_t *d_QuoteBoundaryHeaders_SCAN = 0;  // flags for column boundaries with quotes.
uint32_t *d_printingchars_flags = 0;  // flags for printing chars (first flag non-printing, then reverse the meaning).
uint32_t *d_printingchars_SCAN = 0;  // flags for printing chars (first flag non-printing, then reverse the meaning).
uint8_t *d_CommaHeaders = 0;  // single byte flag for comma headers (comma found, possible col boundary).
uint8_t *d_secondquotes = 0;  // single byte flag for 2nd of "" -- used to flag non-printing.
uint32_t * d_RecordsToQuoteBoundariesTable = 0;

uint16_t *h_ColumnNumInRecord_NAXX = 0;
uint16_t *h_ColumnNumInRecord_a = 0; // aligned


uint32_t *h_UTF8Headers_NAXX = 0;
uint32_t *h_UTF8Headers_a = 0;  // aligned
uint32_t *h_UTF8HeadersSCAN_NAXX = 0;
uint32_t *h_UTF8HeadersSCAN_a = 0;  // aligned
uint32_t *h_UTF8CharsTable_NAXX = 0;
uint32_t *h_UTF8CharsTable_a = 0;  // aligned
uint32_t *h_ColumnsToUTF8charsTable_NAXX = 0;
uint32_t *h_ColumnsToUTF8charsTable_a = 0;  // aligned
uint16_t *h_CharNumInColumn_NAXX = 0;
uint16_t *h_CharNumInColumn_a = 0;  // aligned

uint16_t *d_ColumnNumInRecord = 0;
uint32_t *d_UTF8Headers = 0;
uint32_t *d_UTF8HeadersSCAN = 0;
uint32_t *d_UTF8CharsTable = 0;
uint32_t *d_ColumnsToUTF8charsTable = 0;
uint16_t *d_CharNumInColumn = 0;

uint8_t *  d_CsvBuffer_printing = 0;  // shortened CSV buffer.
uint32_t *  d_RecordHeaders_printing = 0;  // record headers shortened.
uint32_t *  d_ColumnHeaders_printing = 0;  // col headers shortened.

cudaError_t cudaStatus;


int inumchunks = 1;
uint64_t apprx_chunklen = 0;

extern FILE * pCsvFileIn = NULL;
__int64 startseek = (__int64)0;  // place to start in file to read a chunk

uint32_t chunkrecidxstarts[MAXCHUNKS + 1];
uint32_t chunkbufidxstarts[MAXCHUNKS + 1];
uint32_t curchunkreccounts[MAXCHUNKS];
uint32_t curchunkbufbytecounts[MAXCHUNKS];

uint32_t maxchunkreccount;
uint32_t maxchunkbufbytecount;

uint32_t totalrecordscount = 0;  // count of all records, for all chunks

memlinklist_base * h_llbases = NULL;


// temporarily just declare these field values here.
const char *fieldnames[MAXNUMCOLUMNS];
int16_t * h_fieldUTF8charsizes = 0;
uint16_t G_h_fieldbytewidths[MAXNUMCOLUMNS];

unsigned char ** h_fieldptrs = 0;
unsigned char ** h_d_fieldptrs = 0;

int nullterm_addchar = 0;  // add a char to width for a null term (optional).

// count of total valid CSV records.
uint64_t G_totalvalidCSVrecordscount = 0;

// big buffer (used for recombined skus and zips).
char * G_d_strs = 0;

// global for indices (originally a retained sort index, later simple array indices used so index always == ix).
uint64_t * G_d_indices = 0;


char * G_h_d_strings_start_ptr_C[maxsortvals];
uint64_t G_h_d_bytes_per_string_C[maxsortvals];
int G_h_d_sort_defaultup_C[maxsortvals];
bool G_h_d_sortspacetoend_C[maxsortvals];
char G_h_d_sort_datatype_C[maxsortvals];  // the type.
char h_d_values_count_C[1];  // this is just the count of values.  (max is maxsortvals per array sizes above.)


clock_t startTime;
clock_t testTime;
clock_t timePassed;
double secondsPassed;

long G_historyid = 0L;
int validrecordscount = 0;



