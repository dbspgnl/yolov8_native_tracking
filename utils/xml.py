from xml.etree.ElementTree import ElementTree, Element, SubElement

class XML:
    def __init__(self, outPath:str):
        self.outPath = outPath
        self.savedJson = dict()
        self.targetXML = self.inint_xml_file()

    def inint_xml_file(self): # 최초 result.xml 생성
        root = Element("xml")
        root.set("version", "1.0")
        root.set("encoding", "utf-8")
        tracking = Element("tracking")
        root.append(tracking)
        tree = ElementTree(root)
        with open(self.outPath, "wb") as file:
            tree.write(file, encoding='utf-8', xml_declaration=False)
            return file
            
    def make_xml(self):
        def create_tracking_history(json):
            tracking_history = Element("tracking_history")
            tracking_history.set("card", str(json["card"]))
            tracking_history.set("first_appearnce_frame", str(json["first_appearnce_frame"]))
            tracking_history.set("type", str(json["class"]))
            positions = SubElement(tracking_history, "positions")
            positions.set("vector_size", str(json["vector_size"]))
            positions.text = str(json["position"])
            tracking.append(tracking_history)
            
        # result.xml로 데이터 기록
        targetXML = open(self.outPath, 'rt', encoding='UTF8')
        root = ElementTree.parse(ElementTree, source=targetXML)
        tracking = root.find("tracking")
        
        # 해당하는 경우 = 추가 / 모두 해당하지 않는 경우 = 생성
        for k,v in self.savedJson.items():
            if len(tracking) == 0: # 최초 데이터 (이후 실행 안함)
                create_tracking_history(v)
                ElementTree(root).write(self.outPath)
                continue
            tracking_history = tracking.find("tracking_history").find(f"[@card='{k}']")
            if tracking_history == None:
                create_tracking_history(v)
        
        ElementTree(root).write(self.outPath) # 저장
        
    def save_data(self, json):
        if json["card"] not in self.savedJson:
            self.savedJson[json["card"]] = json
            self.savedJson[json["card"]]["first_appearnce_frame"] = json["now_frame"]
            self.savedJson[json["card"]]["vector_size"] = 1
        else: 
            self.savedJson[json["card"]]["position"] += " "+json["position"]
            self.savedJson[json["card"]]["vector_size"] += 1
        
