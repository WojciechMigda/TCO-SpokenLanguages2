#!/bin/bash

MY_DIR=$( dirname ${0} )

MPG123_DIR=$( cd ${MY_DIR}/../mpg123/ && pwd )
INSTALL_DIR=$( cd ${MY_DIR}/../external/ && pwd )

if [ "$(uname)" == "Darwin" ]; then
    export MAKE="make"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    export MAKE="make"
    MPG123_CPU_ARG="--with-cpu=sse"
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" -o "$(expr substr $(uname -s) 1 7)" == "MSYS_NT" ]; then
    export MAKE="mingw32-make"
    BUILD_TYPE_ARG="--build=i686-pc-mingw32"
    MPG123_CPU_ARG="--with-cpu=x86-64"
fi

############################################################
### build mpg123
############################################################

pushd ${MPG123_DIR}

libtoolize --force && \
aclocal-1.14 && \
autoheader && \
automake-1.14 --force-missing --add-missing && \
autoconf

./configure ${BUILD_TYPE_ARG} --prefix=${INSTALL_DIR} --enable-static --with-optimization=3 ${MPG123_CPU_ARG} && ${MAKE} -j 3 && ${MAKE} install-exec install-data

popd
