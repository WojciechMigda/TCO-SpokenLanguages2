################################################################################
#
#  Copyright (c) 2015 Wojciech Migda
#  All rights reserved
#  Distributed under the terms of the Apache 2.0 license
#
################################################################################
#
#  Filename: CMakeLists.txt
#
#  Decription:
#       Build script for C/C++ tools and libraries
#
#  Authors:
#       Wojciech Migda
#
################################################################################
#
#  History:
#  --------
#  Date         Who  Ticket     Description
#  ----------   ---  ---------  ------------------------------------------------
#  2015-09-16   wm              Initial version
#
################################################################################

cmake_minimum_required(VERSION 2.8)

include_directories( ../external/include )
add_library( mp3decoder SHARED mp3decoder.c )
find_library( MPG123_LIB mpg123 "../external/lib/" )
target_link_libraries( mp3decoder ${MPG123_LIB} )
