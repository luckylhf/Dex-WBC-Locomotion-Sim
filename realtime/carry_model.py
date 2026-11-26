"""Load a verified MuJoCo box preset with two compliant grips."""
import hashlib
import json

import mujoco
import numpy as np
from dex_mimic.deployment.native_mujoco import Plant
from gait import bundle, contract

SIDES = ('l','r')
TCP = ('robot/left_tcp_link','robot/right_tcp_link')
WRISTS = tuple(f'robot/wrist_roll_{side}_link' for side in SIDES)
BOX = 'carry/box'
BOX_HALF = np.array([.145,.175,.095])
WALL = .007
GRIP_OFFSETS = np.array([[-.157,.182,-.065],[-.157,-.182,-.065]])
ARM_AXES = ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_pitch','elbow_yaw','wrist_pitch','wrist_roll')
ARM_NAMES = [[f'{axis}_{s}_joint' for axis in ARM_AXES] for s in SIDES]
ARM_ORDER = np.array([[contract['joint_names'].index(n) for n in names] for names in ARM_NAMES])


def joint_ids(m,names):
    return np.array([m.joint('robot/'+name).id for name in names])


def quaternion(matrix):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q,np.ascontiguousarray(matrix).reshape(9))
    return q


def arm_pose(reference, reach=.37, height=-.02):
    m = mujoco.MjModel.from_binary_path(str(bundle/'model.mjb'))
    d = mujoco.MjData(m)
    d.qpos[:7] = [0,0,.95,1,0,0,0]
    d.qpos[m.jnt_qposadr[joint_ids(m,contract['joint_names'])]] = reference
    mujoco.mj_forward(m,d)
    chest = d.xpos[m.body('robot/waist_pitch_link').id].copy()
    center = chest + [reach-GRIP_OFFSETS[0,0],0.,height-GRIP_OFFSETS[0,2]]
    jac = np.zeros((3,m.nv))
    pose,orientations,errors = [],[],[]
    for side,body_name in enumerate(TCP):
        body = m.body(body_name).id
        joints = joint_ids(m,ARM_NAMES[side])
        qa, va = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
        elbow = .16*(1-2*side)
        d.qpos[qa[:4]] = [.08,elbow,0.,-1.55]
        target = center+GRIP_OFFSETS[side]
        for _ in range(45):
            mujoco.mj_kinematics(m,d)
            mujoco.mj_comPos(m,d)
            err = target-d.xpos[body]
            if np.linalg.norm(err) < 1e-5:
                break
            mujoco.mj_jacBody(m,d,jac,None,body)
            J = jac[:,va[:4]]
            dq = J.T@np.linalg.solve(J@J.T+np.eye(3)*1e-4,err)
            d.qpos[qa[:4]] = np.clip(d.qpos[qa[:4]]+np.clip(dq,-.20,.20),
                                     *(m.jnt_range[joints[:4]]+[.01,-.01]).T)
        mujoco.mj_kinematics(m,d)
        pose.append(d.qpos[qa].copy())
        orientations.append(quaternion(d.xmat[body].reshape(3,3).T))
        errors.append(np.linalg.norm(target-d.xpos[body]))
    if max(errors)>.002:
        raise ValueError('Grasp points are unreachable: '+str(errors))
    return np.asarray(pose),orientations


def build_box(mass, orientations, solref=.05, *, added_mass=0., content_mass=0.):
    if content_mass == 0.:
        directory = bundle.parent / 'carry_models'
        catalog = json.loads((directory / 'index.json').read_text())
        if catalog['mujoco_version'] != mujoco.__version__:
            raise RuntimeError('The bundled box models require MuJoCo ' + catalog['mujoco_version'])
        for preset in catalog['models']:
            if (mass == preset['shell_kg'] and added_mass == preset['added_kg']
                    and solref == preset['solref_s']
                    and np.allclose(orientations, preset['orientations'], rtol=0., atol=1e-12)):
                path = directory / preset['file']
                with path.open('rb') as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if actual != preset['sha256']:
                    raise RuntimeError(f'Box model checksum mismatch: {path.name}')
                return mujoco.MjModel.from_binary_path(str(path))

    # The interactive viewer also supports custom shell/content masses. Build those
    # from the same source MJCF and box construction as tools/compile_models.py so
    # the robot remains identical to the frozen baseline.
    from tools.compile_models import build
    model = build(bundle.parents[1], contract, {
        'shell_kg': mass, 'added_kg': added_mass, 'solref_s': solref,
        'orientations': np.asarray(orientations, dtype=float),
        'content_kg': content_mass,
    })
    baseline = mujoco.MjModel.from_binary_path(str(bundle / 'model.mjb'))
    for field in ('body_mass', 'body_ipos', 'jnt_range', 'actuator_gainprm',
                  'actuator_biasprm', 'geom_pos', 'geom_friction', 'dof_damping',
                  'dof_armature'):
        a, b = getattr(baseline, field), getattr(model, field)
        if not np.array_equal(a, b[:len(a)]):
            raise RuntimeError(f'Robot model changed while adding box: {field}')
    return model


class BoxPlant(Plant):
    def __init__(self, controller, model):
        self.controller=controller
        self.model=m=model
        self.data=d=mujoco.MjData(m)
        joints=joint_ids(m,contract['joint_names'])
        self.qadr,self.vadr=m.jnt_qposadr[joints],m.jnt_dofadr[joints]
        self.aids=np.array([int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]) for j in joints])
        root=m.body_jntadr[m.body('robot/pelvis').id]
        self.rq,self.rv=int(m.jnt_qposadr[root]),int(m.jnt_dofadr[root])
        self.box_id=m.body(BOX).id
        free=m.joint('carry/free').id
        self.bq,self.bv=int(m.jnt_qposadr[free]),int(m.jnt_dofadr[free])
        ref=controller.reference
        d.qpos[self.qadr]=ref['q'][0]
        d.qpos[self.rq:self.rq+7]=np.r_[ref['root_p'][0],[1.,0.,0.,0.]]
        d.ctrl[self.aids]=d.qpos[self.qadr]
        mujoco.mj_forward(m,d)
        hand=np.array([d.site(f'carry/hand_{s}').xpos.copy() for s in SIDES])
        d.qpos[self.bq:self.bq+7]=np.r_[(hand-GRIP_OFFSETS).mean(0),[1.,0.,0.,0.]]
        mujoco.mj_forward(m,d)
