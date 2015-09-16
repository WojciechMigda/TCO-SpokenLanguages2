/*******************************************************************************
 * Copyright (c) 2015 Wojciech Migda
 * All rights reserved
 * Distributed under the terms of the Apache 2.0 license
 *******************************************************************************
 *
 * Filename: mp3decoder.c
 *
 * Description:
 *      Very basic MP3 to RAW decoder using MPG123
 *
 * Authors:
 *          Wojciech Migda (wm)
 *
 *******************************************************************************
 * History:
 * --------
 * Date         Who  Ticket     Description
 * ----------   ---  ---------  ------------------------------------------------
 * 2015-09-16   wm              Initial version
 *
 ******************************************************************************/

#include <mpg123.h>

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

unsigned int
mp3decoder(const char * fname, unsigned char * obuf, unsigned int sz)
{
    mpg123_init();

    int err = MPG123_OK;
    mpg123_handle * mh = mpg123_new(NULL, &err);
    assert(mh != NULL && err == MPG123_OK);

    const size_t BUFFER_SZ = mpg123_outblock(mh);

    unsigned char * buffer = (unsigned char *)malloc(BUFFER_SZ);
    assert(buffer != NULL);

    err = mpg123_open(mh, fname);
    assert(err == MPG123_OK);

    long rate = 0;
    int channels = 0;
    int encoding = 0;
    mpg123_getformat(mh, &rate, &channels, &encoding);

    size_t total_read = 0;
    size_t done = 0;
    while ((mpg123_read(mh, buffer, MIN(BUFFER_SZ, sz - total_read), &done) == MPG123_OK) && (total_read < sz))
    {
        memcpy(obuf + total_read, buffer, MIN(BUFFER_SZ, sz - total_read));
        total_read += done;
    }

    free(buffer);
    mpg123_close(mh);
    mpg123_delete(mh);
    mpg123_exit();

    return total_read;
}

#if 0
int main()
{
    unsigned char buf[4000] = {0x5A};

    const unsigned int nread = decoder("../data/train/000kouqjfnk.mp3", buf, sizeof (buf) / sizeof (buf[0]));

    printf("Read %u bytes\n", nread);

    return 0;
}
#endif
