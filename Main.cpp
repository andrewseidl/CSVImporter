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
#include <stdint.h>

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>

#include <stdlib.h>
#include <time.h>

#ifdef WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif

extern "C" uint64_t CSVImporterMain(char * filename, char delimiter, uint16_t numTotalColumns, uint16_t numDefinedColumns, int16_t * ColumnCharWidths, unsigned char ** dataColumnPtrs, unsigned int * dataColumnOffsets, int64_t seekafterhdr, uint8_t charmultiplier, bool GPUResidentFlag);

int main(int argc, char** argv)
{
	char buffer[256];
	memset(buffer, 0, 256);

	// Get the current working directory.
	getcwd(buffer, 256);

	// Add trailing backslash and test filename
	strcat(buffer, "/testfile.csv");

	int16_t SampleColumnCharWidths[64];  // define char widths.
										 // preliminary columns
	int64_t seekafterhdr = 0;

	uint16_t numTotalColumns = 5;  // number of total columns, including those to the right that are not defined.
	uint16_t numDefinedColumns = 4;  // count of Defined Columns (these are ordered from left to right).
	SampleColumnCharWidths[0] = 15;
	SampleColumnCharWidths[1] = -1;
	SampleColumnCharWidths[2] = 8;
	SampleColumnCharWidths[3] = 8;
	char delimiter = '|'; // if separator is a comma else false
	seekafterhdr = 0;  // 0 the first seek.
	uint8_t charmultiplier = 1;	 // multiply by storage multiplier (e.g., 3 for UTF-8, 1 for ASCII)

	unsigned char ** dataColumnPtrs = NULL;
	dataColumnPtrs = (unsigned char **)malloc(numDefinedColumns * sizeof(unsigned char *));
	unsigned int dataColumnOffsets[3]; // Needs to only be as many as the defined columns that we are interested in (ie., their ColumnCharWidths are NOT -1)

	// BELOW IS THE CURRENT GENERIC PROCESS
	clock_t startTime = clock(); //Start timer

	uint64_t x = CSVImporterMain(buffer, delimiter, numTotalColumns, numDefinedColumns, SampleColumnCharWidths, dataColumnPtrs, dataColumnOffsets, seekafterhdr, charmultiplier, false /*GPUResidentFlag*/);

	// BELOW IS THE CURRENT GENERIC PROCESS
	clock_t endTime = clock(); //Ends timer

	printf("Total Time = %f(seconds)\n",(double)(((double)(endTime - startTime))/CLOCKS_PER_SEC));
	printf("%f records per second\n", (double)x / (double)(((double)(endTime - startTime)) / CLOCKS_PER_SEC));
	// If number of columns returned is positive
	if (x > 0)
	{
		// Loop through and write out the records
		for (uint64_t recNo = 0; recNo < x && recNo < 100 /*print UP TO the first 100*/; recNo++)
		{
			// Loop through all the columns in a record
			for (int iLCV = 0; iLCV < 3 /*number of actual columns we want back (ie., width NOT equal -1*/; iLCV++)
			{
				// print the columns to the screen
				char * col = (char *)dataColumnPtrs[iLCV];
				col = (char *)&(col[recNo * dataColumnOffsets[iLCV]]);
				printf("%s", col);
				printf("\t");
			}
			printf("\n");
		}
	}
	return 0;
}
