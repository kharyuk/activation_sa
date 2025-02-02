# Tool for compressing oversized jupyter notebooks
# based on the snippet from
# https://discourse.jupyter.org/t/
# heres-a-script-to-reduce-size-of-a-notebook-with-many-png-images/29964

# Note: python 3.10+ required 

import nbformat
import io
import PIL.Image
import base64

from typing import Any

def convert_png_to_jpg(stream_string: str, jpeg_quality: float = 0.9) -> str:
    png_data = base64.b64decode(stream_string)
    with PIL.Image.open(io.BytesIO(png_data)) as img:
        jpeg_buffer = io.BytesIO()
        # do not convert to RGB as it distorts the text
        img.save(
            jpeg_buffer,
            format="jpeg",
            quality=int(100*jpeg_quality),
            #optimize=True,
            #keep_rgb=True,
            #dpi=(300, 300)
        )
        jpeg_data = jpeg_buffer.getvalue()
    return base64.b64encode(jpeg_data).decode("utf-8")
    
def replace_png_by_jpg(cell_output_dict: dict[str, Any], **kwargs) -> None:
    if (
        not "data" in cell_output_dict
        or not "image/png" in cell_output_dict["data"]
    ):
        return
    jpeg_str: str = convert_png_to_jpg(
        cell_output_dict["data"]["image/png"], **kwargs
    )
    cell_output_dict["data"]["image/jpeg"] = jpeg_str
    del cell_output_dict["data"]["image/png"]
    
def resize_png(stream_string: str, q: float = 0.9) -> str:
    png_data = base64.b64decode(stream_string)
    with PIL.Image.open(io.BytesIO(png_data)) as img:
        img_buffer = io.BytesIO()
        w: int
        h: int
        w, h = img.size
        w = int(w*q)
        h = int(h*q)
        # do not convert to RGB as it distorts the text
        img.resize(
            (w, h), PIL.Image.Resampling.BOX
            # Image.Resampling.NEAREST (0)
            # Image.Resampling.LANCZOS (1)
            # Image.Resampling.BILINEAR (2)
            # Image.Resampling.BICUBIC (3)
            # Image.Resampling.BOX (4)
            # Image.Resampling.HAMMING (5)
        ).save(
            img_buffer, format="png", optimize=True, dpi=(90, 90)
        )
        img_data = img_buffer.getvalue()
    return base64.b64encode(img_data).decode("utf-8")
    
def replace_png_by_resized_png(
    cell_output_dict: dict[str, Any], **kwargs
) -> None:
    if (
        not "data" in cell_output_dict
        or not "image/png" in cell_output_dict["data"]
    ):
        return
    png_str: str = resize_png(
        cell_output_dict["data"]["image/png"], **kwargs
    )
    cell_output_dict["data"]["image/png"] = png_str
    

def compress_ipynb(
    nbfile_path_out: str, nbfile_path_in: str, quality=0.9
) -> None:
    nb: nbformat.notebooknode.NotebookNode
    with open(nbfile_path_in, 'r') as f:
        nb = nbformat.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell["cell_type"] != "code" or len(cell["outputs"]) == 0:
            continue
        for cell_output_dict in cell["outputs"]:
            if cell_output_dict["output_type"] != "display_data":
                continue
            #replace_png_by_jpg(cell_output_dict, jpeg_quality=quality)
            replace_png_by_resized_png(cell_output_dict, q=quality)
    
    with open(nbfile_path_out, 'w') as f:
        nbformat.write(nb, f)
            
if __name__ == "__main__":
    nbfile_path_in: str = "2-2_VGG11_sensitivity_analysis.ipynb"
    nbfile_path_out: str = "comp_2-2_VGG11_sensitivity_analysis.ipynb"
    q: float = 0.95
    compress_ipynb(nbfile_path_out, nbfile_path_in, q)

