from .simple3dposeBaseSMPL import Simple3DPoseBaseSMPL
from .simple3dposeBaseSMPL24 import Simple3DPoseBaseSMPL24
from .simple3dposeSMPLWithCam import Simple3DPoseBaseSMPLCam
from .simple3dposeSMPLWithCamReg import Simple3DPoseBaseSMPLCamReg
from .HRNetWithCam import HRNetSMPLCam
from .HRNetWithCamReg import HRNetSMPLCamReg
from .criterion import *  # noqa: F401,F403

# Optional imports that require PyTorch3D
try:
    from .HRNetSMPLXCamKid import HRNetSMPLXCamKid
    from .HRNetSMPLXCamKidReg import HRNetSMPLXCamKidReg
    _has_smplx = True
except ImportError:
    _has_smplx = False

__all__ = [
    'Simple3DPoseBaseSMPL', 'Simple3DPoseBaseSMPL24', 'Simple3DPoseBaseSMPLCam',
    'Simple3DPoseBaseSMPLCamReg',
    'HRNetSMPLCam', 'HRNetSMPLCamReg']

if _has_smplx:
    __all__.extend(['HRNetSMPLXCamKid', 'HRNetSMPLXCamKidReg'])
