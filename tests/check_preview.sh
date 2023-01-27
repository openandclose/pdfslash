#!/bin/sh

# pdf_reference_1-7.pdf
# (https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/pdfreference1.7old.pdf)
# PDFUA-Ref-2-05_BookChapter-german.pdf
# (one of https://www.pdfa.org/resource/pdfua-reference-suite/)

if [ -n "$1" ]; then  # any argument, for shorter test
    FILE=PDFUA-Ref-2-05_BookChapter-german.pdf
else
    FILE=pdf_reference_1-7.pdf
fi

date +"%Y-%m-%dT%H:%M:%S"
echo 'file:' $FILE
echo 'path:' $(which pdfslash)
pip list | grep numpy
pip list | grep PyMuPDF
pip list | grep pdfslash

pdfslash --_time --_nobanner --command 'preview -_q; exit' $FILE

echo ''
echo ''
echo ''

notify-send 'check_preview -- done'
