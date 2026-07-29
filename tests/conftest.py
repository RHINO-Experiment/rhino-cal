"""Suite-wide configuration.

float64 is enabled here, once, rather than in each test module. The consistency
suite added in a later task compares against a numpy reference at ``rtol=1e-13``;
under JAX's default float32 that comparison cannot pass, and a module that
simply forgot the line would report a plausible-looking failure with no hint
that precision was the cause. Setting it in ``conftest.py`` means no test module
can forget it -- pytest imports this before collecting any of them.
"""

import jax

jax.config.update("jax_enable_x64", True)
