#!/bin/bash
$1 > $1.lastout
diff -q $1.lastout $2
