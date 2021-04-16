#!/bin/bash

set -e
set -x

brew install coreutils ninja
wget https://services.gradle.org/distributions/gradle-6.8.3-all.zip
unzip gradle-6.8.3-all.zip
ln -sf $(pwd)/gradle-6.8.3/bin/gradle /usr/local/bin/gradle
