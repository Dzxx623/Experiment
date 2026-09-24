from pathlib import Path
import xml.etree.ElementTree as ET
xml=Path(__file__).with_name("models")/"planar_arm.xml"
ET.parse(xml)
import mujoco
m=mujoco.MjModel.from_xml_path(str(xml)); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
assert m.nq==3 and m.nu==3
print("MuJoCo XML load OK", "nq",m.nq,"nu",m.nu)
