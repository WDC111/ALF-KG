import os
import lxml.etree as ET
from copy import deepcopy

def get_svg_dimensions(root):
    """从SVG根元素提取宽度、高度和viewBox。返回(width, height, viewBox)"""
    viewBox = root.get('viewBox')
    if viewBox:
        parts = viewBox.strip().split()
        if len(parts) == 4:
            w = float(parts[2])
            h = float(parts[3])
            return w, h, viewBox
    width = root.get('width')
    height = root.get('height')
    if width and height:
        w = float(width.replace('px', '').replace('pt', '').strip())
        h = float(height.replace('px', '').replace('pt', '').strip())
        viewBox = f"0 0 {w} {h}"
        return w, h, viewBox
    # 默认值，防止出错
    return 300, 200, "0 0 300 200"

def merge_svgs(file_list, output_file):
    # 先读取所有SVG，获取实际尺寸，确定最大宽度和高度
    svg_data = []  # 存储每个文件的 (orig_root, orig_w, orig_h, viewBox)
    max_w = 0
    max_h = 0
    parser = ET.XMLParser(remove_blank_text=True)

    for filename in file_list:
        if not os.path.exists(filename):
            print(f"警告：文件 {filename} 不存在，跳过")
            continue
        tree = ET.parse(filename, parser)
        orig_root = tree.getroot()
        w, h, viewBox = get_svg_dimensions(orig_root)
        svg_data.append((orig_root, w, h, viewBox))
        if w > max_w:
            max_w = w
        if h > max_h:
            max_h = h

    if not svg_data:
        print("没有有效的SVG文件")
        return

    cols = 2
    rows = (len(svg_data) + cols - 1) // cols
    total_width = cols * max_w
    total_height = rows * max_h

    # 定义命名空间
    NS = "http://www.w3.org/2000/svg"
    XLINK = "http://www.w3.org/1999/xlink"
    NS_MAP = {None: NS, 'xlink': XLINK}

    new_root = ET.Element(ET.QName(NS, 'svg'), nsmap=NS_MAP)
    new_root.set('width', str(total_width))
    new_root.set('height', str(total_height))
    new_root.set('viewBox', f"0 0 {total_width} {total_height}")
    new_root.set('version', '1.1')

    for idx, (orig_root, orig_w, orig_h, viewBox) in enumerate(svg_data):
        row = idx // cols
        col = idx % cols
        x = col * max_w
        y = row * max_h

        nested_svg = ET.SubElement(new_root, ET.QName(NS, 'svg'))
        nested_svg.set('x', str(x))
        nested_svg.set('y', str(y))
        nested_svg.set('width', str(max_w))
        nested_svg.set('height', str(max_h))
        nested_svg.set('viewBox', viewBox)
        # 保持比例居中，如果希望左上对齐可以改为 'xMinYMin meet'
        nested_svg.set('preserveAspectRatio', 'xMidYMid meet')
        nested_svg.set('overflow', 'visible')

        for child in orig_root:
            nested_svg.append(deepcopy(child))

    tree_out = ET.ElementTree(new_root)
    tree_out.write(output_file, encoding='utf-8', xml_declaration=True, pretty_print=True)
    print(f"合并完成，输出文件：{output_file}，单元格大小：{max_w} x {max_h}")

if __name__ == "__main__":
    files = [
        "dependence_Urea Nitrogen.svg",
        "dependence_Alanine Aminotransferase.svg",
        "dependence_Immunoglobulin G.svg",
        "dependence_Transferrin.svg",
        "dependence_Oxygen.svg",
        "dependence_Fibrinogen, Functional.svg",
        "dependence_Urea Nitrogen, Urine.svg",
        "dependence_Osmolality, Urine.svg",
        "dependence_Protein.svg",
        "dependence_PT.svg"
    ]
    merge_svgs(files, "merged.svg")