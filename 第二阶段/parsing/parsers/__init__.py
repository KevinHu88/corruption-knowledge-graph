"""具体文件类型解析器。"""

from 第二阶段.parsing.parsers.docx_parser import DocxParser
from 第二阶段.parsing.parsers.excel_parser import ExcelParser
from 第二阶段.parsing.parsers.html_parser import HtmlParser
from 第二阶段.parsing.parsers.pdf_parser import PDFParser
from 第二阶段.parsing.parsers.pptx_parser import PptxParser
from 第二阶段.parsing.parsers.txt_parser import TxtParser

__all__ = [
    "DocxParser",
    "ExcelParser",
    "HtmlParser",
    "PDFParser",
    "PptxParser",
    "TxtParser",
]

