# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
from typing import Optional

import flax.linen as nn
import jax

__all__ = ["BaseNeuralVectorField", "BaseRescalingNet"]


class BaseNeuralVectorField(nn.Module, abc.ABC):

  @abc.abstractmethod
  def __call__(
      self,
      t: jax.Array,
      x: jax.Array,
      condition: Optional[jax.Array] = None,
      keys_model: Optional[jax.Array] = None
  ) -> jax.Array:  # noqa: D102):
    pass


class BaseRescalingNet(nn.Module, abc.ABC):

  @abc.abstractmethod
  def __call__(
      self, x: jax.Array, condition: Optional[jax.Array] = None
  ) -> jax.Array:
    pass
