

# for resNet 50:::
'''
name: bn1.beta || shape: (64,) || dtype: float32
name: bn1.gamma || shape: (64,) || dtype: float32
name: bn1.running_mean || shape: (64,) || dtype: float32
name: bn1.running_var || shape: (64,) || dtype: float32
name: conv1.0.weight || shape: (32, 3, 3, 3) || dtype: float32
name: conv1.1.beta || shape: (32,) || dtype: float32
name: conv1.1.gamma || shape: (32,) || dtype: float32
name: conv1.1.running_mean || shape: (32,) || dtype: float32
name: conv1.1.running_var || shape: (32,) || dtype: float32
name: conv1.3.weight || shape: (32, 32, 3, 3) || dtype: float32
name: conv1.4.beta || shape: (32,) || dtype: float32
name: conv1.4.gamma || shape: (32,) || dtype: float32
name: conv1.4.running_mean || shape: (32,) || dtype: float32
name: conv1.4.running_var || shape: (32,) || dtype: float32
name: conv1.6.weight || shape: (64, 32, 3, 3) || dtype: float32
name: fc.bias || shape: (1000,) || dtype: float32
name: fc.weight || shape: (1000, 2048) || dtype: float32
name: layer1.0.bn1.beta || shape: (64,) || dtype: float32
name: layer1.0.bn1.gamma || shape: (64,) || dtype: float32
name: layer1.0.bn1.running_mean || shape: (64,) || dtype: float32
name: layer1.0.bn1.running_var || shape: (64,) || dtype: float32
name: layer1.0.bn2.beta || shape: (64,) || dtype: float32
name: layer1.0.bn2.gamma || shape: (64,) || dtype: float32
name: layer1.0.bn2.running_mean || shape: (64,) || dtype: float32
name: layer1.0.bn2.running_var || shape: (64,) || dtype: float32
name: layer1.0.bn3.beta || shape: (256,) || dtype: float32
name: layer1.0.bn3.gamma || shape: (256,) || dtype: float32
name: layer1.0.bn3.running_mean || shape: (256,) || dtype: float32
name: layer1.0.bn3.running_var || shape: (256,) || dtype: float32
name: layer1.0.conv1.weight || shape: (64, 64, 1, 1) || dtype: float32
name: layer1.0.conv2.weight || shape: (64, 64, 3, 3) || dtype: float32
name: layer1.0.conv3.weight || shape: (256, 64, 1, 1) || dtype: float32
name: layer1.0.downsample.1.weight || shape: (256, 64, 1, 1) || dtype: float32
name: layer1.0.downsample.2.beta || shape: (256,) || dtype: float32
name: layer1.0.downsample.2.gamma || shape: (256,) || dtype: float32
name: layer1.0.downsample.2.running_mean || shape: (256,) || dtype: float32
name: layer1.0.downsample.2.running_var || shape: (256,) || dtype: float32
name: layer1.1.bn1.beta || shape: (64,) || dtype: float32
name: layer1.1.bn1.gamma || shape: (64,) || dtype: float32
name: layer1.1.bn1.running_mean || shape: (64,) || dtype: float32
name: layer1.1.bn1.running_var || shape: (64,) || dtype: float32
name: layer1.1.bn2.beta || shape: (64,) || dtype: float32
name: layer1.1.bn2.gamma || shape: (64,) || dtype: float32
name: layer1.1.bn2.running_mean || shape: (64,) || dtype: float32
name: layer1.1.bn2.running_var || shape: (64,) || dtype: float32
name: layer1.1.bn3.beta || shape: (256,) || dtype: float32
name: layer1.1.bn3.gamma || shape: (256,) || dtype: float32
name: layer1.1.bn3.running_mean || shape: (256,) || dtype: float32
name: layer1.1.bn3.running_var || shape: (256,) || dtype: float32
name: layer1.1.conv1.weight || shape: (64, 256, 1, 1) || dtype: float32
name: layer1.1.conv2.weight || shape: (64, 64, 3, 3) || dtype: float32
name: layer1.1.conv3.weight || shape: (256, 64, 1, 1) || dtype: float32
name: layer1.2.bn1.beta || shape: (64,) || dtype: float32
name: layer1.2.bn1.gamma || shape: (64,) || dtype: float32
name: layer1.2.bn1.running_mean || shape: (64,) || dtype: float32
name: layer1.2.bn1.running_var || shape: (64,) || dtype: float32
name: layer1.2.bn2.beta || shape: (64,) || dtype: float32
name: layer1.2.bn2.gamma || shape: (64,) || dtype: float32
name: layer1.2.bn2.running_mean || shape: (64,) || dtype: float32
name: layer1.2.bn2.running_var || shape: (64,) || dtype: float32
name: layer1.2.bn3.beta || shape: (256,) || dtype: float32
name: layer1.2.bn3.gamma || shape: (256,) || dtype: float32
name: layer1.2.bn3.running_mean || shape: (256,) || dtype: float32
name: layer1.2.bn3.running_var || shape: (256,) || dtype: float32
name: layer1.2.conv1.weight || shape: (64, 256, 1, 1) || dtype: float32
name: layer1.2.conv2.weight || shape: (64, 64, 3, 3) || dtype: float32
name: layer1.2.conv3.weight || shape: (256, 64, 1, 1) || dtype: float32
name: layer2.0.bn1.beta || shape: (128,) || dtype: float32
name: layer2.0.bn1.gamma || shape: (128,) || dtype: float32
name: layer2.0.bn1.running_mean || shape: (128,) || dtype: float32
name: layer2.0.bn1.running_var || shape: (128,) || dtype: float32
name: layer2.0.bn2.beta || shape: (128,) || dtype: float32
name: layer2.0.bn2.gamma || shape: (128,) || dtype: float32
name: layer2.0.bn2.running_mean || shape: (128,) || dtype: float32
name: layer2.0.bn2.running_var || shape: (128,) || dtype: float32
name: layer2.0.bn3.beta || shape: (512,) || dtype: float32
name: layer2.0.bn3.gamma || shape: (512,) || dtype: float32
name: layer2.0.bn3.running_mean || shape: (512,) || dtype: float32
name: layer2.0.bn3.running_var || shape: (512,) || dtype: float32
name: layer2.0.conv1.weight || shape: (128, 256, 1, 1) || dtype: float32
name: layer2.0.conv2.weight || shape: (128, 128, 3, 3) || dtype: float32
name: layer2.0.conv3.weight || shape: (512, 128, 1, 1) || dtype: float32
name: layer2.0.downsample.1.weight || shape: (512, 256, 1, 1) || dtype: float32
name: layer2.0.downsample.2.beta || shape: (512,) || dtype: float32
name: layer2.0.downsample.2.gamma || shape: (512,) || dtype: float32
name: layer2.0.downsample.2.running_mean || shape: (512,) || dtype: float32
name: layer2.0.downsample.2.running_var || shape: (512,) || dtype: float32
name: layer2.1.bn1.beta || shape: (128,) || dtype: float32
name: layer2.1.bn1.gamma || shape: (128,) || dtype: float32
name: layer2.1.bn1.running_mean || shape: (128,) || dtype: float32
name: layer2.1.bn1.running_var || shape: (128,) || dtype: float32
name: layer2.1.bn2.beta || shape: (128,) || dtype: float32
name: layer2.1.bn2.gamma || shape: (128,) || dtype: float32
name: layer2.1.bn2.running_mean || shape: (128,) || dtype: float32
name: layer2.1.bn2.running_var || shape: (128,) || dtype: float32
name: layer2.1.bn3.beta || shape: (512,) || dtype: float32
name: layer2.1.bn3.gamma || shape: (512,) || dtype: float32
name: layer2.1.bn3.running_mean || shape: (512,) || dtype: float32
name: layer2.1.bn3.running_var || shape: (512,) || dtype: float32
name: layer2.1.conv1.weight || shape: (128, 512, 1, 1) || dtype: float32
name: layer2.1.conv2.weight || shape: (128, 128, 3, 3) || dtype: float32
name: layer2.1.conv3.weight || shape: (512, 128, 1, 1) || dtype: float32
name: layer2.2.bn1.beta || shape: (128,) || dtype: float32
name: layer2.2.bn1.gamma || shape: (128,) || dtype: float32
name: layer2.2.bn1.running_mean || shape: (128,) || dtype: float32
name: layer2.2.bn1.running_var || shape: (128,) || dtype: float32
name: layer2.2.bn2.beta || shape: (128,) || dtype: float32
name: layer2.2.bn2.gamma || shape: (128,) || dtype: float32
name: layer2.2.bn2.running_mean || shape: (128,) || dtype: float32
name: layer2.2.bn2.running_var || shape: (128,) || dtype: float32
name: layer2.2.bn3.beta || shape: (512,) || dtype: float32
name: layer2.2.bn3.gamma || shape: (512,) || dtype: float32
name: layer2.2.bn3.running_mean || shape: (512,) || dtype: float32
name: layer2.2.bn3.running_var || shape: (512,) || dtype: float32
name: layer2.2.conv1.weight || shape: (128, 512, 1, 1) || dtype: float32
name: layer2.2.conv2.weight || shape: (128, 128, 3, 3) || dtype: float32
name: layer2.2.conv3.weight || shape: (512, 128, 1, 1) || dtype: float32
name: layer2.3.bn1.beta || shape: (128,) || dtype: float32
name: layer2.3.bn1.gamma || shape: (128,) || dtype: float32
name: layer2.3.bn1.running_mean || shape: (128,) || dtype: float32
name: layer2.3.bn1.running_var || shape: (128,) || dtype: float32
name: layer2.3.bn2.beta || shape: (128,) || dtype: float32
name: layer2.3.bn2.gamma || shape: (128,) || dtype: float32
name: layer2.3.bn2.running_mean || shape: (128,) || dtype: float32
name: layer2.3.bn2.running_var || shape: (128,) || dtype: float32
name: layer2.3.bn3.beta || shape: (512,) || dtype: float32
name: layer2.3.bn3.gamma || shape: (512,) || dtype: float32
name: layer2.3.bn3.running_mean || shape: (512,) || dtype: float32
name: layer2.3.bn3.running_var || shape: (512,) || dtype: float32
name: layer2.3.conv1.weight || shape: (128, 512, 1, 1) || dtype: float32
name: layer2.3.conv2.weight || shape: (128, 128, 3, 3) || dtype: float32
name: layer2.3.conv3.weight || shape: (512, 128, 1, 1) || dtype: float32
name: layer3.0.bn1.beta || shape: (256,) || dtype: float32
name: layer3.0.bn1.gamma || shape: (256,) || dtype: float32
name: layer3.0.bn1.running_mean || shape: (256,) || dtype: float32
name: layer3.0.bn1.running_var || shape: (256,) || dtype: float32
name: layer3.0.bn2.beta || shape: (256,) || dtype: float32
name: layer3.0.bn2.gamma || shape: (256,) || dtype: float32
name: layer3.0.bn2.running_mean || shape: (256,) || dtype: float32
name: layer3.0.bn2.running_var || shape: (256,) || dtype: float32
name: layer3.0.bn3.beta || shape: (1024,) || dtype: float32
name: layer3.0.bn3.gamma || shape: (1024,) || dtype: float32
name: layer3.0.bn3.running_mean || shape: (1024,) || dtype: float32
name: layer3.0.bn3.running_var || shape: (1024,) || dtype: float32
name: layer3.0.conv1.weight || shape: (256, 512, 1, 1) || dtype: float32
name: layer3.0.conv2.weight || shape: (256, 256, 3, 3) || dtype: float32
name: layer3.0.conv3.weight || shape: (1024, 256, 1, 1) || dtype: float32
name: layer3.0.downsample.1.weight || shape: (1024, 512, 1, 1) || dtype: float32
name: layer3.0.downsample.2.beta || shape: (1024,) || dtype: float32
name: layer3.0.downsample.2.gamma || shape: (1024,) || dtype: float32
name: layer3.0.downsample.2.running_mean || shape: (1024,) || dtype: float32
name: layer3.0.downsample.2.running_var || shape: (1024,) || dtype: float32
name: layer3.1.bn1.beta || shape: (256,) || dtype: float32
name: layer3.1.bn1.gamma || shape: (256,) || dtype: float32
name: layer3.1.bn1.running_mean || shape: (256,) || dtype: float32
name: layer3.1.bn1.running_var || shape: (256,) || dtype: float32
name: layer3.1.bn2.beta || shape: (256,) || dtype: float32
name: layer3.1.bn2.gamma || shape: (256,) || dtype: float32
name: layer3.1.bn2.running_mean || shape: (256,) || dtype: float32
name: layer3.1.bn2.running_var || shape: (256,) || dtype: float32
name: layer3.1.bn3.beta || shape: (1024,) || dtype: float32
name: layer3.1.bn3.gamma || shape: (1024,) || dtype: float32
name: layer3.1.bn3.running_mean || shape: (1024,) || dtype: float32
name: layer3.1.bn3.running_var || shape: (1024,) || dtype: float32
name: layer3.1.conv1.weight || shape: (256, 1024, 1, 1) || dtype: float32
name: layer3.1.conv2.weight || shape: (256, 256, 3, 3) || dtype: float32
name: layer3.1.conv3.weight || shape: (1024, 256, 1, 1) || dtype: float32
name: layer3.2.bn1.beta || shape: (256,) || dtype: float32
name: layer3.2.bn1.gamma || shape: (256,) || dtype: float32
name: layer3.2.bn1.running_mean || shape: (256,) || dtype: float32
name: layer3.2.bn1.running_var || shape: (256,) || dtype: float32
name: layer3.2.bn2.beta || shape: (256,) || dtype: float32
name: layer3.2.bn2.gamma || shape: (256,) || dtype: float32
name: layer3.2.bn2.running_mean || shape: (256,) || dtype: float32
name: layer3.2.bn2.running_var || shape: (256,) || dtype: float32
name: layer3.2.bn3.beta || shape: (1024,) || dtype: float32
name: layer3.2.bn3.gamma || shape: (1024,) || dtype: float32
name: layer3.2.bn3.running_mean || shape: (1024,) || dtype: float32
name: layer3.2.bn3.running_var || shape: (1024,) || dtype: float32
name: layer3.2.conv1.weight || shape: (256, 1024, 1, 1) || dtype: float32
name: layer3.2.conv2.weight || shape: (256, 256, 3, 3) || dtype: float32
name: layer3.2.conv3.weight || shape: (1024, 256, 1, 1) || dtype: float32
name: layer3.3.bn1.beta || shape: (256,) || dtype: float32
name: layer3.3.bn1.gamma || shape: (256,) || dtype: float32
name: layer3.3.bn1.running_mean || shape: (256,) || dtype: float32
name: layer3.3.bn1.running_var || shape: (256,) || dtype: float32
name: layer3.3.bn2.beta || shape: (256,) || dtype: float32
name: layer3.3.bn2.gamma || shape: (256,) || dtype: float32
name: layer3.3.bn2.running_mean || shape: (256,) || dtype: float32
name: layer3.3.bn2.running_var || shape: (256,) || dtype: float32
name: layer3.3.bn3.beta || shape: (1024,) || dtype: float32
name: layer3.3.bn3.gamma || shape: (1024,) || dtype: float32
name: layer3.3.bn3.running_mean || shape: (1024,) || dtype: float32
name: layer3.3.bn3.running_var || shape: (1024,) || dtype: float32
name: layer3.3.conv1.weight || shape: (256, 1024, 1, 1) || dtype: float32
name: layer3.3.conv2.weight || shape: (256, 256, 3, 3) || dtype: float32
name: layer3.3.conv3.weight || shape: (1024, 256, 1, 1) || dtype: float32
name: layer3.4.bn1.beta || shape: (256,) || dtype: float32
name: layer3.4.bn1.gamma || shape: (256,) || dtype: float32
name: layer3.4.bn1.running_mean || shape: (256,) || dtype: float32
name: layer3.4.bn1.running_var || shape: (256,) || dtype: float32
name: layer3.4.bn2.beta || shape: (256,) || dtype: float32
name: layer3.4.bn2.gamma || shape: (256,) || dtype: float32
name: layer3.4.bn2.running_mean || shape: (256,) || dtype: float32
name: layer3.4.bn2.running_var || shape: (256,) || dtype: float32
name: layer3.4.bn3.beta || shape: (1024,) || dtype: float32
name: layer3.4.bn3.gamma || shape: (1024,) || dtype: float32
name: layer3.4.bn3.running_mean || shape: (1024,) || dtype: float32
name: layer3.4.bn3.running_var || shape: (1024,) || dtype: float32
name: layer3.4.conv1.weight || shape: (256, 1024, 1, 1) || dtype: float32
name: layer3.4.conv2.weight || shape: (256, 256, 3, 3) || dtype: float32
name: layer3.4.conv3.weight || shape: (1024, 256, 1, 1) || dtype: float32
name: layer3.5.bn1.beta || shape: (256,) || dtype: float32
name: layer3.5.bn1.gamma || shape: (256,) || dtype: float32
name: layer3.5.bn1.running_mean || shape: (256,) || dtype: float32
name: layer3.5.bn1.running_var || shape: (256,) || dtype: float32
name: layer3.5.bn2.beta || shape: (256,) || dtype: float32
name: layer3.5.bn2.gamma || shape: (256,) || dtype: float32
name: layer3.5.bn2.running_mean || shape: (256,) || dtype: float32
name: layer3.5.bn2.running_var || shape: (256,) || dtype: float32
name: layer3.5.bn3.beta || shape: (1024,) || dtype: float32
name: layer3.5.bn3.gamma || shape: (1024,) || dtype: float32
name: layer3.5.bn3.running_mean || shape: (1024,) || dtype: float32
name: layer3.5.bn3.running_var || shape: (1024,) || dtype: float32
name: layer3.5.conv1.weight || shape: (256, 1024, 1, 1) || dtype: float32
name: layer3.5.conv2.weight || shape: (256, 256, 3, 3) || dtype: float32
name: layer3.5.conv3.weight || shape: (1024, 256, 1, 1) || dtype: float32
name: layer4.0.bn1.beta || shape: (512,) || dtype: float32
name: layer4.0.bn1.gamma || shape: (512,) || dtype: float32
name: layer4.0.bn1.running_mean || shape: (512,) || dtype: float32
name: layer4.0.bn1.running_var || shape: (512,) || dtype: float32
name: layer4.0.bn2.beta || shape: (512,) || dtype: float32
name: layer4.0.bn2.gamma || shape: (512,) || dtype: float32
name: layer4.0.bn2.running_mean || shape: (512,) || dtype: float32
name: layer4.0.bn2.running_var || shape: (512,) || dtype: float32
name: layer4.0.bn3.beta || shape: (2048,) || dtype: float32
name: layer4.0.bn3.gamma || shape: (2048,) || dtype: float32
name: layer4.0.bn3.running_mean || shape: (2048,) || dtype: float32
name: layer4.0.bn3.running_var || shape: (2048,) || dtype: float32
name: layer4.0.conv1.weight || shape: (512, 1024, 1, 1) || dtype: float32
name: layer4.0.conv2.weight || shape: (512, 512, 3, 3) || dtype: float32
name: layer4.0.conv3.weight || shape: (2048, 512, 1, 1) || dtype: float32
name: layer4.0.downsample.1.weight || shape: (2048, 1024, 1, 1) || dtype: float32
name: layer4.0.downsample.2.beta || shape: (2048,) || dtype: float32
name: layer4.0.downsample.2.gamma || shape: (2048,) || dtype: float32
name: layer4.0.downsample.2.running_mean || shape: (2048,) || dtype: float32
name: layer4.0.downsample.2.running_var || shape: (2048,) || dtype: float32
name: layer4.1.bn1.beta || shape: (512,) || dtype: float32
name: layer4.1.bn1.gamma || shape: (512,) || dtype: float32
name: layer4.1.bn1.running_mean || shape: (512,) || dtype: float32
name: layer4.1.bn1.running_var || shape: (512,) || dtype: float32
name: layer4.1.bn2.beta || shape: (512,) || dtype: float32
name: layer4.1.bn2.gamma || shape: (512,) || dtype: float32
name: layer4.1.bn2.running_mean || shape: (512,) || dtype: float32
name: layer4.1.bn2.running_var || shape: (512,) || dtype: float32
name: layer4.1.bn3.beta || shape: (2048,) || dtype: float32
name: layer4.1.bn3.gamma || shape: (2048,) || dtype: float32
name: layer4.1.bn3.running_mean || shape: (2048,) || dtype: float32
name: layer4.1.bn3.running_var || shape: (2048,) || dtype: float32
name: layer4.1.conv1.weight || shape: (512, 2048, 1, 1) || dtype: float32
name: layer4.1.conv2.weight || shape: (512, 512, 3, 3) || dtype: float32
name: layer4.1.conv3.weight || shape: (2048, 512, 1, 1) || dtype: float32
name: layer4.2.bn1.beta || shape: (512,) || dtype: float32
name: layer4.2.bn1.gamma || shape: (512,) || dtype: float32
name: layer4.2.bn1.running_mean || shape: (512,) || dtype: float32
name: layer4.2.bn1.running_var || shape: (512,) || dtype: float32
name: layer4.2.bn2.beta || shape: (512,) || dtype: float32
name: layer4.2.bn2.gamma || shape: (512,) || dtype: float32
name: layer4.2.bn2.running_mean || shape: (512,) || dtype: float32
name: layer4.2.bn2.running_var || shape: (512,) || dtype: float32
name: layer4.2.bn3.beta || shape: (2048,) || dtype: float32
name: layer4.2.bn3.gamma || shape: (2048,) || dtype: float32
name: layer4.2.bn3.running_mean || shape: (2048,) || dtype: float32
name: layer4.2.bn3.running_var || shape: (2048,) || dtype: float32
name: layer4.2.conv1.weight || shape: (512, 2048, 1, 1) || dtype: float32
name: layer4.2.conv2.weight || shape: (512, 512, 3, 3) || dtype: float32
name: layer4.2.conv3.weight || shape: (2048, 512, 1, 1) || dtype: float32
'''

'''
C1/conv0/BatchNorm/beta :: conv1.1.beta
C1/conv0/BatchNorm/gamma :: conv1.1.gamma
C1/conv0/BatchNorm/moving_mean :: conv1.1.running_mean
C1/conv0/BatchNorm/moving_variance :: conv1.1.running_var
C1/conv0/weights :: conv1.0.weight
C1/conv1/beta :: conv1.4.beta
C1/conv1/gamma :: conv1.4.gamma
C1/conv1/moving_mean :: conv1.4.running_mean
C1/conv1/moving_variance :: conv1.4.running_var
C1/conv1/weights :: conv1.3.weight
C1/conv2/beta :: bn1.beta
C1/conv2/gamma :: bn1.gamma
C1/conv2/moving_mean :: bn1.running_mean
C1/conv2/moving_variance :: bn1.running_var
C1/conv2/weights :: conv1.6.weight
C2/bottleneck_0/conv0/beta :: layer1.0.bn1.beta
C2/bottleneck_0/conv0/gamma :: layer1.0.bn1.gamma
C2/bottleneck_0/conv0/moving_mean :: layer1.0.bn1.running_mean
C2/bottleneck_0/conv0/moving_variance :: layer1.0.bn1.running_var
C2/bottleneck_0/conv0/weights :: layer1.0.conv1.weight
C2/bottleneck_0/conv1/beta :: layer1.0.bn2.beta
C2/bottleneck_0/conv1/gamma :: layer1.0.bn2.gamma
C2/bottleneck_0/conv1/moving_mean :: layer1.0.bn2.running_mean
C2/bottleneck_0/conv1/moving_variance :: layer1.0.bn2.running_var
C2/bottleneck_0/conv1/weights :: layer1.0.conv2.weight
C2/bottleneck_0/conv2/beta :: layer1.0.bn3.beta
C2/bottleneck_0/conv2/gamma :: layer1.0.bn3.gamma
C2/bottleneck_0/conv2/moving_mean :: layer1.0.bn3.running_mean
C2/bottleneck_0/conv2/moving_variance :: layer1.0.bn3.running_var
C2/bottleneck_0/conv2/weights :: layer1.0.conv3.weight
C2/bottleneck_0/shortcut/beta :: layer1.0.downsample.2.beta
C2/bottleneck_0/shortcut/gamma :: layer1.0.downsample.2.gamma
C2/bottleneck_0/shortcut/moving_mean :: layer1.0.downsample.2.running_mean
C2/bottleneck_0/shortcut/moving_variance :: layer1.0.downsample.2.running_var
C2/bottleneck_0/shortcut/weights :: layer1.0.downsample.1.weight
C2/bottleneck_1/conv0/beta :: layer1.1.bn1.beta
C2/bottleneck_1/conv0/gamma :: layer1.1.bn1.gamma
C2/bottleneck_1/conv0/moving_mean :: layer1.1.bn1.running_mean
C2/bottleneck_1/conv0/moving_variance :: layer1.1.bn1.running_var
C2/bottleneck_1/conv0/weights :: layer1.1.conv1.weight
C2/bottleneck_1/conv1/beta :: layer1.1.bn2.beta
C2/bottleneck_1/conv1/gamma :: layer1.1.bn2.gamma
C2/bottleneck_1/conv1/moving_mean :: layer1.1.bn2.running_mean
C2/bottleneck_1/conv1/moving_variance :: layer1.1.bn2.running_var
C2/bottleneck_1/conv1/weights :: layer1.1.conv2.weight
C2/bottleneck_1/conv2/beta :: layer1.1.bn3.beta
C2/bottleneck_1/conv2/gamma :: layer1.1.bn3.gamma
C2/bottleneck_1/conv2/moving_mean :: layer1.1.bn3.running_mean
C2/bottleneck_1/conv2/moving_variance :: layer1.1.bn3.running_var
C2/bottleneck_1/conv2/weights :: layer1.1.conv3.weight
C2/bottleneck_2/conv0/beta :: layer1.2.bn1.beta
C2/bottleneck_2/conv0/gamma :: layer1.2.bn1.gamma
C2/bottleneck_2/conv0/moving_mean :: layer1.2.bn1.running_mean
C2/bottleneck_2/conv0/moving_variance :: layer1.2.bn1.running_var
C2/bottleneck_2/conv0/weights :: layer1.2.conv1.weight
C2/bottleneck_2/conv1/beta :: layer1.2.bn2.beta
C2/bottleneck_2/conv1/gamma :: layer1.2.bn2.gamma
C2/bottleneck_2/conv1/moving_mean :: layer1.2.bn2.running_mean
C2/bottleneck_2/conv1/moving_variance :: layer1.2.bn2.running_var
C2/bottleneck_2/conv1/weights :: layer1.2.conv2.weight
C2/bottleneck_2/conv2/beta :: layer1.2.bn3.beta
C2/bottleneck_2/conv2/gamma :: layer1.2.bn3.gamma
C2/bottleneck_2/conv2/moving_mean :: layer1.2.bn3.running_mean
C2/bottleneck_2/conv2/moving_variance :: layer1.2.bn3.running_var
C2/bottleneck_2/conv2/weights :: layer1.2.conv3.weight
C3/bottleneck_0/conv0/beta :: layer2.0.bn1.beta
C3/bottleneck_0/conv0/gamma :: layer2.0.bn1.gamma
C3/bottleneck_0/conv0/moving_mean :: layer2.0.bn1.running_mean
C3/bottleneck_0/conv0/moving_variance :: layer2.0.bn1.running_var
C3/bottleneck_0/conv0/weights :: layer2.0.conv1.weight
C3/bottleneck_0/conv1/beta :: layer2.0.bn2.beta
C3/bottleneck_0/conv1/gamma :: layer2.0.bn2.gamma
C3/bottleneck_0/conv1/moving_mean :: layer2.0.bn2.running_mean
C3/bottleneck_0/conv1/moving_variance :: layer2.0.bn2.running_var
C3/bottleneck_0/conv1/weights :: layer2.0.conv2.weight
C3/bottleneck_0/conv2/beta :: layer2.0.bn3.beta
C3/bottleneck_0/conv2/gamma :: layer2.0.bn3.gamma
C3/bottleneck_0/conv2/moving_mean :: layer2.0.bn3.running_mean
C3/bottleneck_0/conv2/moving_variance :: layer2.0.bn3.running_var
C3/bottleneck_0/conv2/weights :: layer2.0.conv3.weight
C3/bottleneck_0/shortcut/beta :: layer2.0.downsample.2.beta
C3/bottleneck_0/shortcut/gamma :: layer2.0.downsample.2.gamma
C3/bottleneck_0/shortcut/moving_mean :: layer2.0.downsample.2.running_mean
C3/bottleneck_0/shortcut/moving_variance :: layer2.0.downsample.2.running_var
C3/bottleneck_0/shortcut/weights :: layer2.0.downsample.1.weight
C3/bottleneck_1/conv0/beta :: layer2.1.bn1.beta
C3/bottleneck_1/conv0/gamma :: layer2.1.bn1.gamma
C3/bottleneck_1/conv0/moving_mean :: layer2.1.bn1.running_mean
C3/bottleneck_1/conv0/moving_variance :: layer2.1.bn1.running_var
C3/bottleneck_1/conv0/weights :: layer2.1.conv1.weight
C3/bottleneck_1/conv1/beta :: layer2.1.bn2.beta
C3/bottleneck_1/conv1/gamma :: layer2.1.bn2.gamma
C3/bottleneck_1/conv1/moving_mean :: layer2.1.bn2.running_mean
C3/bottleneck_1/conv1/moving_variance :: layer2.1.bn2.running_var
C3/bottleneck_1/conv1/weights :: layer2.1.conv2.weight
C3/bottleneck_1/conv2/beta :: layer2.1.bn3.beta
C3/bottleneck_1/conv2/gamma :: layer2.1.bn3.gamma
C3/bottleneck_1/conv2/moving_mean :: layer2.1.bn3.running_mean
C3/bottleneck_1/conv2/moving_variance :: layer2.1.bn3.running_var
C3/bottleneck_1/conv2/weights :: layer2.1.conv3.weight
C3/bottleneck_2/conv0/beta :: layer2.2.bn1.beta
C3/bottleneck_2/conv0/gamma :: layer2.2.bn1.gamma
C3/bottleneck_2/conv0/moving_mean :: layer2.2.bn1.running_mean
C3/bottleneck_2/conv0/moving_variance :: layer2.2.bn1.running_var
C3/bottleneck_2/conv0/weights :: layer2.2.conv1.weight
C3/bottleneck_2/conv1/beta :: layer2.2.bn2.beta
C3/bottleneck_2/conv1/gamma :: layer2.2.bn2.gamma
C3/bottleneck_2/conv1/moving_mean :: layer2.2.bn2.running_mean
C3/bottleneck_2/conv1/moving_variance :: layer2.2.bn2.running_var
C3/bottleneck_2/conv1/weights :: layer2.2.conv2.weight
C3/bottleneck_2/conv2/beta :: layer2.2.bn3.beta
C3/bottleneck_2/conv2/gamma :: layer2.2.bn3.gamma
C3/bottleneck_2/conv2/moving_mean :: layer2.2.bn3.running_mean
C3/bottleneck_2/conv2/moving_variance :: layer2.2.bn3.running_var
C3/bottleneck_2/conv2/weights :: layer2.2.conv3.weight
C3/bottleneck_3/conv0/beta :: layer2.3.bn1.beta
C3/bottleneck_3/conv0/gamma :: layer2.3.bn1.gamma
C3/bottleneck_3/conv0/moving_mean :: layer2.3.bn1.running_mean
C3/bottleneck_3/conv0/moving_variance :: layer2.3.bn1.running_var
C3/bottleneck_3/conv0/weights :: layer2.3.conv1.weight
C3/bottleneck_3/conv1/beta :: layer2.3.bn2.beta
C3/bottleneck_3/conv1/gamma :: layer2.3.bn2.gamma
C3/bottleneck_3/conv1/moving_mean :: layer2.3.bn2.running_mean
C3/bottleneck_3/conv1/moving_variance :: layer2.3.bn2.running_var
C3/bottleneck_3/conv1/weights :: layer2.3.conv2.weight
C3/bottleneck_3/conv2/beta :: layer2.3.bn3.beta
C3/bottleneck_3/conv2/gamma :: layer2.3.bn3.gamma
C3/bottleneck_3/conv2/moving_mean :: layer2.3.bn3.running_mean
C3/bottleneck_3/conv2/moving_variance :: layer2.3.bn3.running_var
C3/bottleneck_3/conv2/weights :: layer2.3.conv3.weight
C4/bottleneck_0/conv0/beta :: layer3.0.bn1.beta
C4/bottleneck_0/conv0/gamma :: layer3.0.bn1.gamma
C4/bottleneck_0/conv0/moving_mean :: layer3.0.bn1.running_mean
C4/bottleneck_0/conv0/moving_variance :: layer3.0.bn1.running_var
C4/bottleneck_0/conv0/weights :: layer3.0.conv1.weight
C4/bottleneck_0/conv1/beta :: layer3.0.bn2.beta
C4/bottleneck_0/conv1/gamma :: layer3.0.bn2.gamma
C4/bottleneck_0/conv1/moving_mean :: layer3.0.bn2.running_mean
C4/bottleneck_0/conv1/moving_variance :: layer3.0.bn2.running_var
C4/bottleneck_0/conv1/weights :: layer3.0.conv2.weight
C4/bottleneck_0/conv2/beta :: layer3.0.bn3.beta
C4/bottleneck_0/conv2/gamma :: layer3.0.bn3.gamma
C4/bottleneck_0/conv2/moving_mean :: layer3.0.bn3.running_mean
C4/bottleneck_0/conv2/moving_variance :: layer3.0.bn3.running_var
C4/bottleneck_0/conv2/weights :: layer3.0.conv3.weight
C4/bottleneck_0/shortcut/beta :: layer3.0.downsample.2.beta
C4/bottleneck_0/shortcut/gamma :: layer3.0.downsample.2.gamma
C4/bottleneck_0/shortcut/moving_mean :: layer3.0.downsample.2.running_mean
C4/bottleneck_0/shortcut/moving_variance :: layer3.0.downsample.2.running_var
C4/bottleneck_0/shortcut/weights :: layer3.0.downsample.1.weight
C4/bottleneck_1/conv0/beta :: layer3.1.bn1.beta
C4/bottleneck_1/conv0/gamma :: layer3.1.bn1.gamma
C4/bottleneck_1/conv0/moving_mean :: layer3.1.bn1.running_mean
C4/bottleneck_1/conv0/moving_variance :: layer3.1.bn1.running_var
C4/bottleneck_1/conv0/weights :: layer3.1.conv1.weight
C4/bottleneck_1/conv1/beta :: layer3.1.bn2.beta
C4/bottleneck_1/conv1/gamma :: layer3.1.bn2.gamma
C4/bottleneck_1/conv1/moving_mean :: layer3.1.bn2.running_mean
C4/bottleneck_1/conv1/moving_variance :: layer3.1.bn2.running_var
C4/bottleneck_1/conv1/weights :: layer3.1.conv2.weight
C4/bottleneck_1/conv2/beta :: layer3.1.bn3.beta
C4/bottleneck_1/conv2/gamma :: layer3.1.bn3.gamma
C4/bottleneck_1/conv2/moving_mean :: layer3.1.bn3.running_mean
C4/bottleneck_1/conv2/moving_variance :: layer3.1.bn3.running_var
C4/bottleneck_1/conv2/weights :: layer3.1.conv3.weight
C4/bottleneck_2/conv0/beta :: layer3.2.bn1.beta
C4/bottleneck_2/conv0/gamma :: layer3.2.bn1.gamma
C4/bottleneck_2/conv0/moving_mean :: layer3.2.bn1.running_mean
C4/bottleneck_2/conv0/moving_variance :: layer3.2.bn1.running_var
C4/bottleneck_2/conv0/weights :: layer3.2.conv1.weight
C4/bottleneck_2/conv1/beta :: layer3.2.bn2.beta
C4/bottleneck_2/conv1/gamma :: layer3.2.bn2.gamma
C4/bottleneck_2/conv1/moving_mean :: layer3.2.bn2.running_mean
C4/bottleneck_2/conv1/moving_variance :: layer3.2.bn2.running_var
C4/bottleneck_2/conv1/weights :: layer3.2.conv2.weight
C4/bottleneck_2/conv2/beta :: layer3.2.bn3.beta
C4/bottleneck_2/conv2/gamma :: layer3.2.bn3.gamma
C4/bottleneck_2/conv2/moving_mean :: layer3.2.bn3.running_mean
C4/bottleneck_2/conv2/moving_variance :: layer3.2.bn3.running_var
C4/bottleneck_2/conv2/weights :: layer3.2.conv3.weight
C4/bottleneck_3/conv0/beta :: layer3.3.bn1.beta
C4/bottleneck_3/conv0/gamma :: layer3.3.bn1.gamma
C4/bottleneck_3/conv0/moving_mean :: layer3.3.bn1.running_mean
C4/bottleneck_3/conv0/moving_variance :: layer3.3.bn1.running_var
C4/bottleneck_3/conv0/weights :: layer3.3.conv1.weight
C4/bottleneck_3/conv1/beta :: layer3.3.bn2.beta
C4/bottleneck_3/conv1/gamma :: layer3.3.bn2.gamma
C4/bottleneck_3/conv1/moving_mean :: layer3.3.bn2.running_mean
C4/bottleneck_3/conv1/moving_variance :: layer3.3.bn2.running_var
C4/bottleneck_3/conv1/weights :: layer3.3.conv2.weight
C4/bottleneck_3/conv2/beta :: layer3.3.bn3.beta
C4/bottleneck_3/conv2/gamma :: layer3.3.bn3.gamma
C4/bottleneck_3/conv2/moving_mean :: layer3.3.bn3.running_mean
C4/bottleneck_3/conv2/moving_variance :: layer3.3.bn3.running_var
C4/bottleneck_3/conv2/weights :: layer3.3.conv3.weight
C4/bottleneck_4/conv0/beta :: layer3.4.bn1.beta
C4/bottleneck_4/conv0/gamma :: layer3.4.bn1.gamma
C4/bottleneck_4/conv0/moving_mean :: layer3.4.bn1.running_mean
C4/bottleneck_4/conv0/moving_variance :: layer3.4.bn1.running_var
C4/bottleneck_4/conv0/weights :: layer3.4.conv1.weight
C4/bottleneck_4/conv1/beta :: layer3.4.bn2.beta
C4/bottleneck_4/conv1/gamma :: layer3.4.bn2.gamma
C4/bottleneck_4/conv1/moving_mean :: layer3.4.bn2.running_mean
C4/bottleneck_4/conv1/moving_variance :: layer3.4.bn2.running_var
C4/bottleneck_4/conv1/weights :: layer3.4.conv2.weight
C4/bottleneck_4/conv2/beta :: layer3.4.bn3.beta
C4/bottleneck_4/conv2/gamma :: layer3.4.bn3.gamma
C4/bottleneck_4/conv2/moving_mean :: layer3.4.bn3.running_mean
C4/bottleneck_4/conv2/moving_variance :: layer3.4.bn3.running_var
C4/bottleneck_4/conv2/weights :: layer3.4.conv3.weight
C4/bottleneck_5/conv0/beta :: layer3.5.bn1.beta
C4/bottleneck_5/conv0/gamma :: layer3.5.bn1.gamma
C4/bottleneck_5/conv0/moving_mean :: layer3.5.bn1.running_mean
C4/bottleneck_5/conv0/moving_variance :: layer3.5.bn1.running_var
C4/bottleneck_5/conv0/weights :: layer3.5.conv1.weight
C4/bottleneck_5/conv1/beta :: layer3.5.bn2.beta
C4/bottleneck_5/conv1/gamma :: layer3.5.bn2.gamma
C4/bottleneck_5/conv1/moving_mean :: layer3.5.bn2.running_mean
C4/bottleneck_5/conv1/moving_variance :: layer3.5.bn2.running_var
C4/bottleneck_5/conv1/weights :: layer3.5.conv2.weight
C4/bottleneck_5/conv2/beta :: layer3.5.bn3.beta
C4/bottleneck_5/conv2/gamma :: layer3.5.bn3.gamma
C4/bottleneck_5/conv2/moving_mean :: layer3.5.bn3.running_mean
C4/bottleneck_5/conv2/moving_variance :: layer3.5.bn3.running_var
C4/bottleneck_5/conv2/weights :: layer3.5.conv3.weight
C5/bottleneck_0/conv0/beta :: layer4.0.bn1.beta
C5/bottleneck_0/conv0/gamma :: layer4.0.bn1.gamma
C5/bottleneck_0/conv0/moving_mean :: layer4.0.bn1.running_mean
C5/bottleneck_0/conv0/moving_variance :: layer4.0.bn1.running_var
C5/bottleneck_0/conv0/weights :: layer4.0.conv1.weight
C5/bottleneck_0/conv1/beta :: layer4.0.bn2.beta
C5/bottleneck_0/conv1/gamma :: layer4.0.bn2.gamma
C5/bottleneck_0/conv1/moving_mean :: layer4.0.bn2.running_mean
C5/bottleneck_0/conv1/moving_variance :: layer4.0.bn2.running_var
C5/bottleneck_0/conv1/weights :: layer4.0.conv2.weight
C5/bottleneck_0/conv2/beta :: layer4.0.bn3.beta
C5/bottleneck_0/conv2/gamma :: layer4.0.bn3.gamma
C5/bottleneck_0/conv2/moving_mean :: layer4.0.bn3.running_mean
C5/bottleneck_0/conv2/moving_variance :: layer4.0.bn3.running_var
C5/bottleneck_0/conv2/weights :: layer4.0.conv3.weight
C5/bottleneck_0/shortcut/beta :: layer4.0.downsample.2.beta
C5/bottleneck_0/shortcut/gamma :: layer4.0.downsample.2.gamma
C5/bottleneck_0/shortcut/moving_mean :: layer4.0.downsample.2.running_mean
C5/bottleneck_0/shortcut/moving_variance :: layer4.0.downsample.2.running_var
C5/bottleneck_0/shortcut/weights :: layer4.0.downsample.1.weight
C5/bottleneck_1/conv0/beta :: layer4.1.bn1.beta
C5/bottleneck_1/conv0/gamma :: layer4.1.bn1.gamma
C5/bottleneck_1/conv0/moving_mean :: layer4.1.bn1.running_mean
C5/bottleneck_1/conv0/moving_variance :: layer4.1.bn1.running_var
C5/bottleneck_1/conv0/weights :: layer4.1.conv1.weight
C5/bottleneck_1/conv1/beta :: layer4.1.bn2.beta
C5/bottleneck_1/conv1/gamma :: layer4.1.bn2.gamma
C5/bottleneck_1/conv1/moving_mean :: layer4.1.bn2.running_mean
C5/bottleneck_1/conv1/moving_variance :: layer4.1.bn2.running_var
C5/bottleneck_1/conv1/weights :: layer4.1.conv2.weight
C5/bottleneck_1/conv2/beta :: layer4.1.bn3.beta
C5/bottleneck_1/conv2/gamma :: layer4.1.bn3.gamma
C5/bottleneck_1/conv2/moving_mean :: layer4.1.bn3.running_mean
C5/bottleneck_1/conv2/moving_variance :: layer4.1.bn3.running_var
C5/bottleneck_1/conv2/weights :: layer4.1.conv3.weight
C5/bottleneck_2/conv0/beta :: layer4.2.bn1.beta
C5/bottleneck_2/conv0/gamma :: layer4.2.bn1.gamma
C5/bottleneck_2/conv0/moving_mean :: layer4.2.bn1.running_mean
C5/bottleneck_2/conv0/moving_variance :: layer4.2.bn1.running_var
C5/bottleneck_2/conv0/weights :: layer4.2.conv1.weight
C5/bottleneck_2/conv1/beta :: layer4.2.bn2.beta
C5/bottleneck_2/conv1/gamma :: layer4.2.bn2.gamma
C5/bottleneck_2/conv1/moving_mean :: layer4.2.bn2.running_mean
C5/bottleneck_2/conv1/moving_variance :: layer4.2.bn2.running_var
C5/bottleneck_2/conv1/weights :: layer4.2.conv2.weight
C5/bottleneck_2/conv2/beta :: layer4.2.bn3.beta
C5/bottleneck_2/conv2/gamma :: layer4.2.bn3.gamma
C5/bottleneck_2/conv2/moving_mean :: layer4.2.bn3.running_mean
C5/bottleneck_2/conv2/moving_variance :: layer4.2.bn3.running_var
C5/bottleneck_2/conv2/weights :: layer4.2.conv3.weight
logits/biases :: fc.bias
logits/weights :: fc.weight
'''