"""Build the MPX dynamics model from the URDF actually published by ROS."""
from types import SimpleNamespace
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def model_from_description(description, config):
    """Return a floating-base model and a private config derived from that model.

    ROS/Gazebo extensions and visual-only geometry are not dynamics inputs.
    Inertias, collisions, joint transforms, axes and limits come from the URDF.
    No fallback to the standalone MPX XML is allowed here.
    """
    root = ET.fromstring(description)
    if root.tag != 'robot':
        raise ValueError('robot_description must be an expanded URDF <robot>')
    initial = {}
    effort_limits = {}
    for joint in root.findall('ros2_control/joint'):
        value = joint.find('state_interface[@name="position"]/param[@name="initial_value"]')
        if value is not None:
            initial[joint.get('name')] = float(value.text)
    for joint in root.findall('joint'):
        limit = joint.find('limit')
        if limit is not None and limit.get('effort') is not None:
            effort_limits[joint.get('name')] = float(limit.get('effort'))
    children = {j.find('child').get('link') for j in root.findall('joint')}
    roots = [link.get('name') for link in root.findall('link') if link.get('name') not in children]
    if roots != [config.base_body_name]:
        raise ValueError(f'Expected root {config.base_body_name}, got {roots}')
    for element in list(root):
        if element.tag not in ('link', 'joint'):
            root.remove(element)
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            link.remove(visual)
        for index, collision in enumerate(link.findall('collision')):
            suffix = f'_{index}' if index else ''
            collision.set('name', f'{link.get("name")}_collision{suffix}')
    for mesh in root.findall('.//mesh'):
        uri = urlparse(mesh.get('filename'))
        if uri.scheme == 'file':
            mesh.set('filename', unquote(uri.path))
        elif uri.scheme == 'package':
            from ament_index_python.packages import get_package_share_directory
            mesh.set('filename', get_package_share_directory(uri.netloc) + unquote(uri.path))
        elif uri.scheme or not mesh.get('filename').startswith('/'):
            raise ValueError('Collision meshes must use absolute paths or file/package URIs')
    ET.SubElement(root, 'link', name='world')
    joint = ET.SubElement(root, 'joint', name='mpx_floating_base', type='floating')
    ET.SubElement(joint, 'parent', link='world')
    ET.SubElement(joint, 'child', link=config.base_body_name)
    extension = ET.SubElement(root, 'mujoco')
    ET.SubElement(extension, 'compiler', fusestatic='false', discardvisual='true',
                  inertiafromgeom='false', strippath='false')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    # MPX computes contact forces in its WBC; Gazebo owns collision simulation.
    # Keep the URDF shapes for contact kinematics, without MJX collision pairs.
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    names = [model.joint(i).name for i in range(1, model.njnt)]
    if names != list(config.joints_name) or model.nq != 15 or model.nv != 14:
        raise ValueError(f'MPX requires a floating base and its eight ordered joints; got {names}')
    for kind, names in ((mujoco.mjtObj.mjOBJ_BODY, [config.base_body_name, *config.body_name]),
                        (mujoco.mjtObj.mjOBJ_GEOM, config.contact_frame)):
        for name in names:
            if mujoco.mj_name2id(model, kind, name) < 0:
                raise ValueError(f'Missing MPX model element: {name}')
    contacts = [model.geom(name).id for name in config.contact_frame]
    if any(model.geom_type[i] != mujoco.mjtGeom.mjGEOM_CYLINDER for i in contacts):
        raise ValueError('MPX wheel contacts must be cylinders')
    radii = model.geom_size[contacts, 0]
    if not np.allclose(radii, radii[0]):
        raise ValueError('MPX requires equal wheel radii')
    missing = set(config.joints_name) - initial.keys()
    if missing:
        raise ValueError(f'Missing ros2_control initial joint positions: {sorted(missing)}')
    runtime = SimpleNamespace(**{k: v for k, v in vars(config).items() if not k.startswith('__')})
    runtime.model_path = None  # The ROS node must never load the standalone XML.
    runtime.q0 = np.array([initial[name] for name in config.joints_name])
    missing_effort = set(config.joints_name) - effort_limits.keys()
    if missing_effort:
        raise ValueError(f'Missing URDF effort limits: {sorted(missing_effort)}')
    runtime.effort_limits = np.array([effort_limits[name] for name in config.joints_name])
    runtime.mass = float(model.body_mass.sum())
    runtime.wheel_radius = float(radii[0])
    model.opt.gravity[:] = [0, 0, -config.grav]
    model.opt.timestep = config.dt_sim
    data = mujoco.MjData(model)
    data.qpos[7:] = runtime.q0
    mujoco.mj_forward(model, data)
    runtime.d = float(np.linalg.norm(data.geom_xpos[contacts[0]] - data.geom_xpos[contacts[1]]))
    runtime.grf_ref = np.tile([0., 0., config.grav * runtime.mass / config.n_contact], config.n_contact)
    runtime.u_ref = np.concatenate([np.zeros(3), runtime.grf_ref])
    return model, runtime
