# Test-only shim for the reference container: the same narrow torch safe-globals
# allowlist the worker uses, so upstream whisperx can load its bundled VAD checkpoint.
import collections, typing
import omegaconf, torch
from pyannote.audio.core.model import Introspection
from pyannote.audio.core.task import Problem, Resolution, Specifications
torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig, omegaconf.base.ContainerMetadata, omegaconf.nodes.AnyNode,
    omegaconf.base.Metadata, typing.Any, list, dict, int, collections.defaultdict,
    torch.torch_version.TorchVersion, Introspection, Specifications, Problem, Resolution,
])
