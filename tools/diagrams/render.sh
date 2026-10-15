#!/bin/bash
# render.sh file.svg out.png bg width
f=$(realpath "$1"); out=$2; bg=$3; w=${4:-1000}
html=/tmp/r_$$.html
echo "<html><body style='margin:0;padding:20px;background:$bg'><img src='file://$f' style='width:$((w-40))px;display:block'></body></html>" > $html
google-chrome --headless=new --no-sandbox --disable-gpu --hide-scrollbars --window-size=$w,${5:-900} --screenshot=$out file://$html >/dev/null 2>&1
rm $html
