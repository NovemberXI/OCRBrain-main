import os
import json
import fitz # PyMuPDF
from PIL import Image
from utils.LogTool import LogTool

class FileTool:
    @staticmethod
    def ensureDir(filePath):
        """
        确保文件所在的目录存在
        """
        directory = os.path.dirname(filePath)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception as e:
                LogTool.error(f"Failed to create directory: {directory}", e)

    @staticmethod
    def appendJsonLine(filePath, dataDict):
        """
        向文件追加一行 JSON 数据 (JSONL 格式)
        """
        try:
            FileTool.ensureDir(filePath)
            with open(filePath, 'a', encoding='utf-8') as f:
                jsonLine = json.dumps(dataDict, ensure_ascii=False)
                f.write(jsonLine + "\n")
            return True
        except Exception as e:
            LogTool.error(f"Failed to append to file: {filePath}", e)
            return False

    @staticmethod
    def pdfToImage(pdfPath, dpi=300):
        """
        将PDF文件的每一页转换为PIL Image对象列表。
        Args:
            pdfPath (str): PDF文件的路径。
            dpi (int): 渲染图像的分辨率。
        Returns:
            list: 包含每个PDF页面PIL Image对象的列表。
        """
        images = []
        try:
            document = fitz.open(pdfPath)
            for pageNumber in range(document.page_count):
                page = document.load_page(pageNumber)
                # Render page to an image
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            document.close()
        except Exception as e:
            LogTool.error(f"Failed to convert PDF to image for {pdfPath}: {e}")
        return images

