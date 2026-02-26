# @title Load model

import jax
import jax.numpy as jnp
from videoprism import models as vp

MODEL_NAME = 'videoprism_public_v1_large'  # @param ['videoprism_public_v1_base', 'videoprism_public_v1_large'] {allow-input: false}
USE_BFLOAT16 = False  # @param { type: "boolean" }
NUM_FRAMES = 16
FRAME_SIZE = 288

fprop_dtype = jnp.bfloat16 if USE_BFLOAT16 else None
flax_model = vp.get_model(MODEL_NAME, fprop_dtype=fprop_dtype)
loaded_state = vp.load_pretrained_weights(MODEL_NAME)


@jax.jit
def forward_fn(inputs, train=False):
  return flax_model.apply(loaded_state, inputs, train=train)