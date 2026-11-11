"""Model of FedDistill-Eye: a ViT-Large/16 backbone and a hierarchical classification head.

The backbone is the self-supervised pretrained ViT-Large (best_model.pth). It is frozen at
every institution and is never trained in the framework. The global head is the only part
that the framework learns. It is distilled at the server from the features and soft labels
that the institutions upload.
"""
import timm
import torch
import torch.nn as nn

NUM_CLASSES = 39      # unified label space (the 39 JSIEC classes)
FEATURE_DIM = 1024    # dimension of the backbone feature (CLS token)


def make_head(in_dim=FEATURE_DIM, num_classes=NUM_CLASSES):
    """Hierarchical head: 1024 -> 2048 -> 1024 -> 39."""
    return nn.Sequential(
        nn.Dropout(0.3), nn.LayerNorm(in_dim), nn.Linear(in_dim, 2048), nn.GELU(),
        nn.Dropout(0.3), nn.LayerNorm(2048), nn.Linear(2048, 1024), nn.GELU(),
        nn.Dropout(0.3), nn.LayerNorm(1024), nn.Linear(1024, num_classes))


def build_model():
    """ViT-Large/16 (224 x 224 input) with the hierarchical head."""
    model = timm.create_model('vit_large_patch16_224', pretrained=False)
    model.head = make_head(model.head.in_features)
    return model


def _load(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    return state


def load_backbone(model, path):
    """Load the frozen backbone. The head of the pretrained file is not used."""
    own = model.state_dict()
    state = _load(path)
    kept = {k: v for k, v in state.items() if k in own and not k.startswith('head')}
    missing = [k for k in own if not k.startswith('head') and k not in kept]
    if missing:
        raise RuntimeError('backbone file lacks %d parameters, e.g. %s' % (len(missing), missing[:3]))
    own.update(kept)
    model.load_state_dict(own)
    return model


def load_head(path):
    """Load one global head (a file global_head_seed*.pth)."""
    head = make_head()
    head.load_state_dict(_load(path), strict=True)
    return head


def load_full_model(path):
    """Load a single-file checkpoint that holds the backbone and one head."""
    model = build_model()
    model.load_state_dict(_load(path), strict=True)
    return model


@torch.inference_mode()
def backbone_features(model, images):
    """Feature that an institution uploads: the CLS token after the last layer norm."""
    return model.forward_features(images)[:, 0]
