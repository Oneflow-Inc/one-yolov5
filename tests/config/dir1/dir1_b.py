# refer to https://github.com/facebookresearch/detectron2
# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from onedet.config import LazyConfig

# equivalent to relative import
dir1a_str, dir1a_dict = LazyConfig.load_rel("dir1_a.py", ("dir1a_str", "dir1a_dict"))

dir1b_str = dir1a_str + "_from_b"
dir1b_dict = dir1a_dict

# Every import is a reload: not modified by other config files
assert dir1a_dict.a == 1