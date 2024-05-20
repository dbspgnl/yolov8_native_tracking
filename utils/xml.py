from xml.etree.ElementTree import Element, dump, ElementTree

def make_xml(json):
    
    # 만약에 파일이 없다면 result.xml로 양식을 만들기

    targetXML = open("data/result.xml", 'rt', encoding='UTF8')
    root = ElementTree.parse(ElementTree, source=targetXML)

    node_tracking = root.find("tracking")
    
    for history in node_tracking:
        if history.get("type") == "truck":
            # print(history.get("type"))
            position = history.find("positions")
            position.text = position.text + " 1 2 3 4" # 임시 삽입
            # print(position.text)
            
        
    indent(root)
    # dump(root)
    write_xml(root)

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def write_xml(root):
    ElementTree(root).write("data/result.xml")
    
make_xml(None)
