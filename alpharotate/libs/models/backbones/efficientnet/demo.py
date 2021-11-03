# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Contains definitions for EfficientNet model.
[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""
import collections
import math
import numpy as np
import six
import tensorflow as tf
import re
import os

# tf.enable_eager_execution()
l = tf.keras.layers

# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate',
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def relu_fn(x):
    return tf.keras.layers.Lambda(tf.nn.swish)(x)


def drop_connect(inputs, is_training, drop_connect_rate):
    """Apply drop connect."""
    if not is_training:
        return inputs

    # Compute keep_prob
    # TODO(tanmingxing): add support for training progress.
    keep_prob = 1.0 - drop_connect_rate

    # Compute drop_connect tensor
    batch_size = tf.shape(inputs)[0]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.div(inputs, keep_prob) * binary_tensor
    return output


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.
    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused
    Returns:
      an initialization for the variable
    """
    del partition_info
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random_normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for dense kernels.
    This initialization is equal to
      tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                      distribution='uniform').
    It is written out explicitly here for clarity.
    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused
    Returns:
      an initialization for the variable
    """
    del partition_info
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    orig_f = filters
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    tf.logging.info('round_filter input={} output={}'.format(orig_f, new_filters))
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def _bn_layer(axis, momentum, epsilon, name=None):
    return l.BatchNormalization(axis=axis,
                                momentum=momentum,
                                epsilon=epsilon,
                                scale=True,
                                center=True,
                                name=name)


class MBConvBlock(object):
    """A class of MBConv: Mobile Inveretd Residual Bottleneck.
    Attributes:
      has_se: boolean. Whether the block contains a Squeeze and Excitation layer
        inside.
      endpoints: dict. A list of internal tensors.
    """

    def __init__(self, block_args, global_params):
        """Initializes a MBConv block.
        Args:
          block_args: BlockArgs, arguments to create a Block.
          global_params: GlobalParams, a set of global parameters.
        """
        self._block_args = block_args
        self._batch_norm_momentum = global_params.batch_norm_momentum
        self._batch_norm_epsilon = global_params.batch_norm_epsilon
        if global_params.data_format == 'channels_first':
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
        self.has_se = (self._block_args.se_ratio is not None) and (
            self._block_args.se_ratio > 0) and (self._block_args.se_ratio <= 1)

        self.endpoints = None

        # Builds the block accordings to arguments.
        self._build()

    def block_args(self):
        return self._block_args

    def _build(self):
        """Builds block according to the arguments."""
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = l.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=False)
            self._bn0 = _bn_layer(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon)

        kernel_size = self._block_args.kernel_size
        # Depth-wise convolution phase:
        self._depthwise_conv = l.DepthwiseConv2D(
            kernel_size=[kernel_size, kernel_size],
            strides=self._block_args.strides,
            depthwise_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn1 = _bn_layer(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)

        if self.has_se:
            num_reduced_filters = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio))
            # Squeeze and Excitation layer.
            self._se_reduce = l.Conv2D(
                num_reduced_filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=True)
            self._se_expand = l.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=True)

        # Output phase:
        filters = self._block_args.output_filters
        self._project_conv = l.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn2 = _bn_layer(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)

    def _call_se(self, input_tensor):
        """Call Squeeze and Excitation layer.
        Args:
          input_tensor: Tensor, a single input tensor for Squeeze/Excitation layer.
        Returns:
          A output tensor, which should have the same shape as input.
        """
        se_tensor = tf.reduce_mean(input_tensor, self._spatial_dims, keepdims=True)
        se_tensor = self._se_expand(relu_fn(self._se_reduce(se_tensor)))
        tf.logging.info('Built Squeeze and Excitation with tensor shape: %s' %
                        (se_tensor.shape))
        return tf.sigmoid(se_tensor) * input_tensor

    def call(self, inputs, training=True, drop_connect_rate=None, output_layer_name=None):
        """Implementation of call().
        Args:
          inputs: the inputs tensor.
          training: boolean, whether the model is constructed for training.
          drop_connect_rate: float, between 0 to 1, drop connect rate.
          output_layer_name: layer name for output layer
        Returns:
          A output tensor.
        """
        tf.logging.info('Block input: %s shape: %s' % (inputs.name, inputs.shape))
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs), training=training))
        else:
            x = inputs
        tf.logging.info('Expand: %s shape: %s' % (x.name, x.shape))

        x = relu_fn(self._bn1(self._depthwise_conv(x), training=training))
        tf.logging.info('DWConv: %s shape: %s' % (x.name, x.shape))

        if self.has_se:
            with tf.variable_scope('se'):
                x = tf.keras.layers.Lambda(self._call_se)(x)

        if self._block_args.id_skip:
            if all(
                            s == 1 for s in self._block_args.strides
            ) and self._block_args.input_filters == self._block_args.output_filters:
                # only apply drop_connect if skip presents.
                x = self._bn2(self._project_conv(x), training=training)
                if drop_connect_rate:
                    x = tf.keras.layers.Lambda(drop_connect)(x, training, drop_connect_rate)
                x = l.Add(name=output_layer_name)([x, inputs])
            else:
                x = _bn_layer(axis=self._channel_axis,
                              momentum=self._batch_norm_momentum,
                              epsilon=self._batch_norm_epsilon,
                              name=output_layer_name)(self._project_conv(x), training=training)
        else:
            x = _bn_layer(axis=self._channel_axis,
                          momentum=self._batch_norm_momentum,
                          epsilon=self._batch_norm_epsilon,
                          name=output_layer_name)(self._project_conv(x), training=training)
        tf.logging.info('Project: %s shape: %s' % (x.name, x.shape))
        return x


class EfficientNet(tf.keras.Model):
    """A class implements tf.keras.Model for MNAS-like model.
      Reference: https://arxiv.org/abs/1807.11626
    """

    def __init__(self, blocks_args=None, global_params=None):
        """Initializes an `Model` instance.
        Args:
          blocks_args: A list of BlockArgs to construct block modules.
          global_params: GlobalParams, a set of global parameters.
        Raises:
          ValueError: when blocks_args is not specified as a list.
        """
        super(EfficientNet, self).__init__()
        if not isinstance(blocks_args, list):
            raise ValueError('blocks_args should be a list.')
        self._global_params = global_params
        self._blocks_args = blocks_args
        self.endpoints = None
        self._build()

    def _build(self):
        """Builds a model."""
        self._blocks = []
        # Builds blocks.
        for block_args in self._blocks_args:
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters,
                                            self._global_params),
                output_filters=round_filters(block_args.output_filters,
                                             self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params))

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                # pylint: disable=protected-access
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1])
                # pylint: enable=protected-access
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        batch_norm_momentum = self._global_params.batch_norm_momentum
        batch_norm_epsilon = self._global_params.batch_norm_epsilon
        if self._global_params.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        # Stem part.
        self._conv_stem = l.Conv2D(
            filters=round_filters(32, self._global_params),
            kernel_size=[3, 3],
            strides=[2, 2],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False,
            name="stem_conv")
        self._bn0 = _bn_layer(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)

        # Head part.
        self._conv_head = l.Conv2D(
            filters=round_filters(1280, self._global_params),
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn1 = _bn_layer(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)

        self._avg_pooling = l.GlobalAveragePooling2D(
            data_format=self._global_params.data_format)
        self._fc = l.Dense(
            self._global_params.num_classes,
            kernel_initializer=dense_kernel_initializer)

        if self._global_params.dropout_rate > 0:
            self._dropout = l.Dropout(self._global_params.dropout_rate)
        else:
            self._dropout = None

    def call(self, inputs, training=True, features_only=None):
        """Implementation of call().
        Args:
          inputs: input tensors.
          training: boolean, whether the model is constructed for training.
          features_only: build the base feature network only.
        Returns:
          output tensors.
        """
        outputs = None
        self.endpoints = {}
        # Calls Stem layers
        with tf.variable_scope('stem'):
            outputs = relu_fn(
                self._bn0(self._conv_stem(inputs), training=training))
        tf.logging.info('Built stem layers with output shape: %s' % outputs.shape)
        self.endpoints['stem'] = outputs

        # Calls blocks.
        reduction_idx = 0
        for idx, block in enumerate(self._blocks):
            is_reduction = False
            if ((idx == len(self._blocks) - 1) or
                        self._blocks[idx + 1].block_args().strides[0] > 1):
                is_reduction = True
                reduction_idx += 1

            with tf.variable_scope('blocks_%s' % idx):
                drop_rate = self._global_params.drop_connect_rate
                if drop_rate:
                    drop_rate *= float(idx) / len(self._blocks)
                    tf.logging.info('block_%s drop_connect_rate: %s' % (idx, drop_rate))
                outputs = block.call(outputs, training=training, output_layer_name='block_%s' % idx)
                self.endpoints['block_%s' % idx] = outputs
                if is_reduction:
                    self.endpoints['reduction_%s' % reduction_idx] = outputs
                if block.endpoints:
                    for k, v in six.iteritems(block.endpoints):
                        self.endpoints['block_%s/%s' % (idx, k)] = v
                        if is_reduction:
                            self.endpoints['reduction_%s/%s' % (reduction_idx, k)] = v
        self.endpoints['global_pool'] = outputs
        if not features_only:
            # Calls final layers and returns logits.
            with tf.variable_scope('head'):
                outputs = relu_fn(
                    self._bn1(self._conv_head(outputs), training=training))
                outputs = self._avg_pooling(outputs)
                if self._dropout:
                    outputs = self._dropout(outputs, training=training)
                outputs = self._fc(outputs)
                self.endpoints['head'] = outputs
        return outputs

    def call_model(self, inputs, training=True, features_only=None):
        """Implementation of call().
        Args:
          inputs: input tensors.
          training: boolean, whether the model is constructed for training.
          features_only: build the base feature network only.
        Returns:
          output tensors.
        """
        outputs = None
        self.endpoints = {}
        # Calls Stem layers
        with tf.variable_scope('stem'):
            outputs = relu_fn(
                self._bn0(self._conv_stem(inputs), training=training))
        tf.logging.info('Built stem layers with output shape: %s' % outputs.shape)

        # Calls blocks.
        reduction_idx = 0
        for idx, block in enumerate(self._blocks):
            is_reduction = False
            if ((idx == len(self._blocks) - 1) or
                        self._blocks[idx + 1].block_args().strides[0] > 1):
                is_reduction = True
                reduction_idx += 1

            with tf.variable_scope('blocks_%s' % idx):
                drop_rate = self._global_params.drop_connect_rate
                if drop_rate:
                    drop_rate *= float(idx) / len(self._blocks)
                    tf.logging.info('block_%s drop_connect_rate: %s' % (idx, drop_rate))
                outputs = block.call(outputs, training=training, output_layer_name='block_%s' % idx)

        if not features_only:
            # Calls final layers and returns logits.
            with tf.variable_scope('head'):
                outputs = relu_fn(
                    self._bn1(self._conv_head(outputs), training=training))
                outputs = self._avg_pooling(outputs)
                if self._dropout:
                    outputs = self._dropout(outputs, training=training)
                outputs = self._fc(outputs)
        # model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


def efficientnet_params(model_name):
    """Get efficientnet params based on model name."""
    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'noisy_student_efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    """Block Decoder for readability."""

    def _decode_block_string(self, block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            strides=[int(options['s'][0]), int(options['s'][1])])

    def _encode_block_string(self, block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if block.se_ratio > 0 and block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    def decode(self, string_list):
        """Decodes a list of string notations to specify blocks inside the network.
        Args:
          string_list: a list of strings, each string is a notation of block.
        Returns:
          A list of namedtuples to represent blocks arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(self._decode_block_string(block_string))
        return blocks_args

    def encode(self, blocks_args):
        """Encodes a list of Blocks to a list of strings.
        Args:
          blocks_args: A list of namedtuples to represent blocks arguments.
        Returns:
          a list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(self._encode_block_string(block))
        return block_strings


def efficientnet_args(width_coefficient=None,
                      depth_coefficient=None,
                      dropout_rate=0.2,
                      drop_connect_rate=0.2):
    """Creates a efficientnet model args."""
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        data_format='channels_last',
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None)
    decoder = BlockDecoder()
    return decoder.decode(blocks_args), global_params


def get_model_params(model_name, override_params):
    """Get the block args and global params for a given model."""
    if model_name.startswith('efficientnet'):
        width_coefficient, depth_coefficient, _, dropout_rate = (
            efficientnet_params(model_name))
        blocks_args, global_params = efficientnet_args(
            width_coefficient, depth_coefficient, dropout_rate)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)

    if override_params:
        # ValueError will be raised here if override_params has fields not included
        # in global_params.
        global_params = global_params._replace(**override_params)

    tf.logging.info('global_params= %s', global_params)
    tf.logging.info('blocks_args= %s', blocks_args)
    return blocks_args, global_params


def build_model(images,
                model_name,
                training,
                override_params=None,
                model_dir=None):
    """A helper functiion to creates a model and returns predicted logits.
    Args:
      images: input images tensor.
      model_name: string, the predefined model name.
      training: boolean, whether the model is constructed for training.
      override_params: A dictionary of params for overriding. Fields must exist in
        efficientnet_model.GlobalParams.
      model_dir: string, optional model dir for saving configs.
    Returns:
      logits: the logits tensor of classes.
      endpoints: the endpoints for each layer.
    Raises:
      When model_name specified an undefined model, raises NotImplementedError.
      When override_params has invalid fields, raises ValueError.
    """
    assert isinstance(images, tf.Tensor)
    blocks_args, global_params = get_model_params(model_name, override_params)

    if model_dir:
        param_file = os.path.join(model_dir, 'model_params.txt')
        if not tf.gfile.Exists(param_file):
            with tf.gfile.GFile(param_file, 'w') as f:
                tf.logging.info('writing to %s' % param_file)
                f.write('model_name= %s\n\n' % model_name)
                f.write('global_params= %s\n\n' % str(global_params))
                f.write('blocks_args= %s\n\n' % str(blocks_args))

    with tf.variable_scope(model_name):
        model = EfficientNet(blocks_args, global_params)
        logits = model(images, training=training)

    logits = tf.identity(logits, 'logits')
    return logits, model.endpoints


def build_model_base(images, model_name, training, override_params=None):
    """A helper functiion to create a base model and return global_pool.
    Args:
      images: input images tensor.
      model_name: string, the model name of a pre-defined MnasNet.
      training: boolean, whether the model is constructed for training.
      override_params: A dictionary of params for overriding. Fields must exist in
        mnasnet_model.GlobalParams.
    Returns:
      features: global pool features.
      endpoints: the endpoints for each layer.
    Raises:
      When model_name specified an undefined model, raises NotImplementedError.
      When override_params has invalid fields, raises ValueError.
    """
    assert isinstance(images, tf.Tensor)
    blocks_args, global_params = get_model_params(model_name, override_params)

    with tf.variable_scope(model_name):
        model = EfficientNet(blocks_args, global_params)
        features = model(images, training=training, features_only=True)

    # features = tf.identity(features, 'global_pool')
    return features, model.endpoints


def build_model_base_keras_model(input_shape, model_name, training, override_params=None):
    """A helper functiion to create a base model and return global_pool.
    Args:
      images: input images tensor.
      model_name: string, the model name of a pre-defined MnasNet.
      training: boolean, whether the model is constructed for training.
      override_params: A dictionary of params for overriding. Fields must exist in
        mnasnet_model.GlobalParams.
    Returns:
      features: global pool features.
      endpoints: the endpoints for each layer.
    Raises:
      When model_name specified an undefined model, raises NotImplementedError.
      When override_params has invalid fields, raises ValueError.
    """
    blocks_args, global_params = get_model_params(model_name, override_params)

    with tf.variable_scope(model_name):
        inputs = tf.keras.layers.Input(shape=input_shape)
        model = EfficientNet(blocks_args, global_params)
        net = model.call_model(inputs, training=training, features_only=True)
        return net


def restore_model(sess, ckpt_dir):
    """Restore variables from checkpoint dir."""
    checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    # ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    # ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
    # for v in tf.global_variables():
    #     if 'moving_mean' in v.name or 'moving_variance' in v.name:
    #         ema_vars.append(v)
    # ema_vars = list(set(ema_vars))
    # var_dict = ema.variables_to_restore(ema_vars)
    saver = tf.train.Saver(max_to_keep=1)
    saver.restore(sess, checkpoint)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    inputs = tf.ones((1, 640, 640, 3), dtype=tf.float32)
    input_shape = [640, 640, 3]
    model_name = "efficientnet-b0"
    features, endpoints = build_model_base(inputs, model_name, training=True)
    # model = build_model_base_keras_model(input_shape, model_name, True)
    # print(model.get_layer("block_15").output)
    for k, v in endpoints.items():
        print(k, v)

        # b0: feature map keys: ["block_4", "block_10", "block_15"] (1/8, 1/16, 1/32)
        # b1: feature map keys: ["block_7", "block_15", "block_22"]
        # b2: feature map keys: ["block_7", "block_15", "block_22"]
        # b3: feature map keys: ["block_7", "block_17", "block_25"]
        # b4: feature map keys: ["block_9", "block_21", "block_31"]
        # b5: feature map keys: ["block_12", "block_26", "block_38"]
        # b6: feature map keys: ["block_14", "block_30", "block_44"]
        # b7: feature map keys: ["block_17", "block_37", "block_54"]

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    # saver = tf.train.Saver(max_to_keep=5)
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        sess.run(init_op)
        restore_model(sess, '/data/yangxue/code/R3Det_Tensorflow/libs/networks/efficientnet/efficientnet-b0')
        # saver.restore(sess, '/data/yangxue/code/R3Det_Tensorflow/libs/networks/efficientnet/efficientnet-b0/model.ckpt')

        features_, endpoints_ = sess.run([features, endpoints])
        print(features_.shape)