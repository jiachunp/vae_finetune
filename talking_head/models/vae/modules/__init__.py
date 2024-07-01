from .encoders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "talking_head.models.vae.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
