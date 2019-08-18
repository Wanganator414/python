from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
images = convert_from_path('C:\Users\ericw\Desktop\Images\git_cheat_sheet.pdf')
