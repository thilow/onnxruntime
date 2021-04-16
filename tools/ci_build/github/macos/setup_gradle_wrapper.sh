#!/bin/bash

set -e
set -x

pushd java
gradle wrapper --gradle-version 6.8.3
popd

ls -la java
