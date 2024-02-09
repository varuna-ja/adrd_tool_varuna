from . import nn
from . import model
from . import shap
from .model import DynamicCalibratedClassifier
from .model import StaticCalibratedClassifier

# load fitted transformer and calibrated wrapper
try:
    fitted_resnet3d = model.CNNResNet3DWithLinearClassifier.from_ckpt('{}/ckpt/ckpt_img_072523.pt'.format(__path__[0]))
    fitted_calibrated_classifier_nonimg = StaticCalibratedClassifier.from_ckpt(
        filepath_state_dict = '{}/ckpt/static_calibrated_classifier_073023.pkl'.format(__path__[0]),
        filepath_wrapped_model = '{}/ckpt/ckpt_080823.pt'.format(__path__[0]),
    )
    fitted_transformer = fitted_calibrated_classifier_nonimg.model
    shap_explainer = shap.MCExplainer(fitted_transformer)
except:
    print('Fail to load checkpoints.')