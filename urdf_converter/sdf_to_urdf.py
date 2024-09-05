'''
240612

- sdf파일을 urdf파일로 변환하기 위한 코드
- 변경사향_v1 : GoogleScannedObject 폴더 내의 더들에 대해서 자동으로 변환 진행
- 변경사항_v1 : 줄바꿈 적용
- 변경사항_v2 : texture.png를 meshes폴더로 이동하는 코드 추가
'''

import os
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom

def sdf_to_urdf(sdf_path, urdf_path):
    # SDF 파일을 파싱합
    tree = ET.parse(sdf_path)
    sdf_root = tree.getroot()

    # URDF 루트 요소를 생성
    model_name = sdf_root.find('model').get('name')
    urdf_root = ET.Element('robot')
    urdf_root.set('name', model_name)

    # SDF에서 링크를 URDF로 변환
    for sdf_link in sdf_root.findall('.//link'):
        urdf_link = ET.SubElement(urdf_root, 'link')
        urdf_link.set('name', sdf_link.get('name'))

        # 기본적인 inertial 요소 추가
        urdf_inertial = ET.SubElement(urdf_link, 'inertial')
        urdf_origin_inertial = ET.SubElement(urdf_inertial, 'origin')
        urdf_origin_inertial.set('xyz', '0 0 0')
        urdf_origin_inertial.set('rpy', '0 0 0')

        urdf_mass = ET.SubElement(urdf_inertial, 'mass')
        urdf_mass.set('value', '1.0')

        urdf_inertia = ET.SubElement(urdf_inertial, 'inertia')
        urdf_inertia.set('ixx', '0.1')
        urdf_inertia.set('ixy', '0.0')
        urdf_inertia.set('ixz', '0.0')
        urdf_inertia.set('iyy', '0.1')
        urdf_inertia.set('iyz', '0.0')
        urdf_inertia.set('izz', '0.1')

        # Visual 요소 변환
        for sdf_visual in sdf_link.findall('visual'):
            urdf_visual = ET.SubElement(urdf_link, 'visual')
            urdf_visual.set('name', sdf_visual.get('name'))

            urdf_origin_visual = ET.SubElement(urdf_visual, 'origin')
            urdf_origin_visual.set('xyz', '0 0 0')
            urdf_origin_visual.set('rpy', '0 0 0')

            sdf_geometry = sdf_visual.find('geometry')
            urdf_geometry = ET.SubElement(urdf_visual, 'geometry')
            sdf_mesh = sdf_geometry.find('mesh')
            urdf_mesh = ET.SubElement(urdf_geometry, 'mesh')
            urdf_mesh.set('filename', os.path.join(os.path.dirname(sdf_path), sdf_mesh.find('uri').text))

            # urdf_material = ET.SubElement(urdf_visual, 'material')
            # urdf_material.set('name', 'white')
            # urdf_color = ET.SubElement(urdf_material, 'color')
            # urdf_color.set('rgba', '1 1 1 1')

        # Collision 요소 변환
        for sdf_collision in sdf_link.findall('collision'):
            urdf_collision = ET.SubElement(urdf_link, 'collision')
            urdf_collision.set('name', sdf_collision.get('name'))

            urdf_origin_collision = ET.SubElement(urdf_collision, 'origin')
            urdf_origin_collision.set('xyz', '0 0 0')
            urdf_origin_collision.set('rpy', '0 0 0')

            sdf_geometry = sdf_collision.find('geometry')
            urdf_geometry = ET.SubElement(urdf_collision, 'geometry')
            sdf_mesh = sdf_geometry.find('mesh')
            urdf_mesh = ET.SubElement(urdf_geometry, 'mesh')
            print(sdf_mesh.find('uri').text)
            urdf_mesh.set('filename', os.path.join(os.path.dirname(sdf_path), sdf_mesh.find('uri').text))

    # 변환된 URDF 파일을 저장
    rough_string = ET.tostring(urdf_root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_string = reparsed.toprettyxml(indent="  ")

    with open(urdf_path, 'w') as f:
        f.write(pretty_string)

    # 텍스처 파일 이동
    source_texture_path = os.path.join(os.path.dirname(sdf_path), 'materials', 'textures', 'texture.png')
    destination_texture_path = os.path.join(os.path.dirname(sdf_path), 'meshes', 'texture.png')

    if os.path.exists(source_texture_path):
        shutil.move(source_texture_path, destination_texture_path)
        print(f'Moved {source_texture_path} to {destination_texture_path}')


def batch_convert_sdf_to_urdf(root_folder):
    for subdir, _, files in os.walk(root_folder):
        print(files)
        for i in range(len(files)):
            print(f'{i}/{len(files)}')
            file = files[i]
            if file.endswith('.sdf'):
                sdf_path = os.path.join(subdir, file)
                urdf_path = os.path.join(subdir, file.replace('.sdf', '.urdf'))
                sdf_to_urdf(sdf_path, urdf_path)
                print(f'Converted {sdf_path} to {urdf_path}')

# 예제 사용법
root_folder = '/home/kimsy/isaacgym/IsaacGymEnvs/NIA_for_sample_dataset/code/urdf/google/CookingBench'
# batch_convert_sdf_to_urdf(root_folder)
sdf_to_urdf('/home/kimsy/isaacgym/IsaacGymEnvs/NIA_for_sample_dataset/code/urdf/kitchen/nesquik_box/model.sdf', 
            '/home/kimsy/isaacgym/IsaacGymEnvs/NIA_for_sample_dataset/code/urdf/kitchen/nesquik_box/model.urdf')