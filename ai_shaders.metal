// Completing the random_crop kernel
kernel void random_crop(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       device const uint2* cropOffsets [[buffer(2)]],
                       uint3 id [[thread_position_in_grid]],
                       constant uint& batchSize [[buffer(3)]],
                       constant uint& channels [[buffer(4)]],
                       constant uint& inputHeight [[buffer(5)]],
                       constant uint& inputWidth [[buffer(6)]],
                       constant uint& outputHeight [[buffer(7)]],
                       constant uint& outputWidth [[buffer(8)]]) {
    uint x = id.x;
    uint y = id.y;
    uint z = id.z;
    uint batch = z / channels;
    uint channel = z % channels;
    
    if (x >= outputWidth || y >= outputHeight || batch >= batchSize) {
        return;
    }
    
    uint2 offset = cropOffsets[batch];
    uint inputX = x + offset.x;
    uint inputY = y + offset.y;
    
    if (inputX < inputWidth && inputY < inputHeight) {
        uint outputIdx = (batch * channels * outputHeight * outputWidth) +
                         (channel * outputHeight * outputWidth) +
                         (y * outputWidth) + x;
        
        uint inputIdx = (batch * channels * inputHeight * inputWidth) +
                        (channel * inputHeight * inputWidth) +
                        (inputY * inputWidth) + inputX;
        
        output[outputIdx] = input[inputIdx];
    }
}

// Additional data augmentation kernels
kernel void random_flip_horizontal(device const float* input [[buffer(0)]],
                                  device float* output [[buffer(1)]],
                                  device const bool* shouldFlip [[buffer(2)]],
                                  uint3 id [[thread_position_in_grid]],
                                  constant uint& batchSize [[buffer(3)]],
                                  constant uint& channels [[buffer(4)]],
                                  constant uint& height [[buffer(5)]],
                                  constant uint& width [[buffer(6)]]) {
    uint x = id.x;
    uint y = id.y;
    uint z = id.z;
    uint batch = z / channels;
    uint channel = z % channels;
    
    if (x >= width || y >= height || batch >= batchSize) {
        return;
    }
    
    uint outputIdx = (batch * channels * height * width) +
                     (channel * height * width) +
                     (y * width) + x;
    
    if (shouldFlip[batch]) {
        uint flippedX = width - 1 - x;
        uint inputIdx = (batch * channels * height * width) +
                        (channel * height * width) +
                        (y * width) + flippedX;
        output[outputIdx] = input[inputIdx];
    } else {
        output[outputIdx] = input[outputIdx];
    }
}

kernel void random_flip_vertical(device const float* input [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                device const bool* shouldFlip [[buffer(2)]],
                                uint3 id [[thread_position_in_grid]],
                                constant uint& batchSize [[buffer(3)]],
                                constant uint& channels [[buffer(4)]],
                                constant uint& height [[buffer(5)]],
                                constant uint& width [[buffer(6)]]) {
    uint x = id.x;
    uint y = id.y;
    uint z = id.z;
    uint batch = z / channels;
    uint channel = z % channels;
    
    if (x >= width || y >= height || batch >= batchSize) {
        return;
    }
    
    uint outputIdx = (batch * channels * height * width) +
                     (channel * height * width) +
                     (y * width) + x;
    
    if (shouldFlip[batch]) {
        uint flippedY = height - 1 - y;
        uint inputIdx = (batch * channels * height * width) +
                        (channel * height * width) +
                        (flippedY * width) + x;
        output[outputIdx] = input[inputIdx];
    } else {
        output[outputIdx] = input[outputIdx];
    }
}

kernel void random_brightness(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             device const float* factors [[buffer(2)]],
                             uint3 id [[thread_position_in_grid]],
                             constant uint& batchSize [[buffer(3)]],
                             constant uint& channels [[buffer(4)]],
                             constant uint& height [[buffer(5)]],
                             constant uint& width [[buffer(6)]]) {
    uint x = id.x;
    uint y = id.y;
    uint z = id.z;
    uint batch = z / channels;
    uint channel = z % channels;
    
    if (x >= width || y >= height || batch >= batchSize) {
        return;
    }
    
    uint idx = (batch * channels * height * width) +
               (channel * height * width) +
               (y * width) + x;
    
    // Only apply brightness adjustment to the luminance or all channels depending on image format
    output[idx] = input[idx] * factors[batch];
}

kernel void random_contrast(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           device const float* factors [[buffer(2)]],
                           device const float* means [[buffer(3)]],
                           uint3 id [[thread_position_in_grid]],
                           constant uint& batchSize [[buffer(3)]],
                           constant uint& channels [[buffer(4)]],
                           constant uint& height [[buffer(5)]],
                           constant uint& width [[buffer(6)]]) {
    uint x = id.x;
    uint y = id.y;
    uint z = id.z;
    uint batch = z / channels;
    uint channel = z % channels;
    
    if (x >= width || y >= height || batch >= batchSize) {
        return;
    }
    
    uint idx = (batch * channels * height * width) +
               (channel * height * width) +
               (y * width) + x;
    
    float mean = means[batch * channels + channel];
    float factor = factors[batch];
    
    output[idx] = factor * (input[idx] - mean) + mean;
}

kernel void random_rotation(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           device const float* angles [[buffer(2)]],
                           uint3 id [[thread_position_in_grid]],
                           constant uint& batchSize [[buffer(3)]],
                           constant uint& channels [[buffer(4)]],
                           constant uint& height [[buffer(5)]],
                           constant uint& width [[buffer(6)]]) {
    uint x = id.x;
    uint y = id.y;
    uint z = id.z;
    uint batch = z / channels;
    uint channel = z % channels;
    
    if (x >= width || y >= height || batch >= batchSize) {
        return;
    }
    
    float angle = angles[batch];
    float sinAngle = sin(angle);
    float cosAngle = cos(angle);
    
    // Calculate the center of the image
    float centerX = width / 2.0;
    float centerY = height / 2.0;
    
    // Calculate the rotated coordinates
    float xOffset = x - centerX;
    float yOffset = y - centerY;
    
    float rotatedX = cosAngle * xOffset - sinAngle * yOffset + centerX;
    float rotatedY = sinAngle * xOffset + cosAngle * yOffset + centerY;
    
    uint outputIdx = (batch * channels * height * width) +
                     (channel * height * width) +
                     (y * width) + x;
    
    // Use bilinear interpolation for smoother results
    if (rotatedX >= 0 && rotatedX < width - 1 && rotatedY >= 0 && rotatedY < height - 1) {
        uint x0 = uint(rotatedX);
        uint y0 = uint(rotatedY);
        uint x1 = x0 + 1;
        uint y1 = y0 + 1;
        
        float wx = rotatedX - float(x0);
        float wy = rotatedY - float(y0);
        
        uint idx00 = (batch * channels * height * width) + (channel * height * width) + (y0 * width) + x0;
        uint idx01 = (batch * channels * height * width) + (channel * height * width) + (y1 * width) + x0;
        uint idx10 = (batch * channels * height * width) + (channel * height * width) + (y0 * width) + x1;
        uint idx11 = (batch * channels * height * width) + (channel * height * width) + (y1 * width) + x1;
        
        float value = (1.0 - wx) * (1.0 - wy) * input[idx00] +
                      (1.0 - wx) * wy * input[idx01] +
                      wx * (1.0 - wy) * input[idx10] +
                      wx * wy * input[idx11];
        
        output[outputIdx] = value;
    } else {
        // Set to zero or a padding value for out of bounds
        output[outputIdx] = 0.0;
    }
}

kernel void cutout(device const float* input [[buffer(0)]],
                  device float* output [[buffer(1)]],
                  device const uint4* cutoutParams [[buffer(2)]], // x, y, width, height
                  uint3 id [[thread_position_in_grid]],
                  constant uint& batchSize [[buffer(3)]],
                  constant uint& channels [[buffer(4)]],
                  constant uint& height [[buffer(5)]],
                  constant uint& width [[buffer(6)]],
                  constant float& fillValue [[buffer(7)]]) {
    uint x = id.x;
    uint y = id.y;
    uint z = id.z;
    uint batch = z / channels;
    uint channel = z % channels;
    
    if (x >= width || y >= height || batch >= batchSize) {
        return;
    }
    
    uint idx = (batch * channels * height * width) +
               (channel * height * width) +
               (y * width) + x;
    
    uint4 params = cutoutParams[batch];
    uint cutX = params.x;
    uint cutY = params.y;
    uint cutWidth = params.z;
    uint cutHeight = params.w;
    
    if (x >= cutX && x < cutX + cutWidth && y >= cutY && y < cutY + cutHeight) {
        output[idx] = fillValue;
    } else {
        output[idx] = input[idx];
    }
}

kernel void mixup(device const float* inputA [[buffer(0)]],
                 device const float* inputB [[buffer(1)]],
                 device float* output [[buffer(2)]],
                 device const float* mixingFactors [[buffer(3)]],
                 uint3 id [[thread_position_in_grid]],
                 constant uint& batchSize [[buffer(4)]],
                 constant uint& channels [[buffer(5)]],
                 constant uint& height [[buffer(6)]],
                 constant uint& width [[buffer(7)]]) {
    uint x = id.x;
    uint y = id.y;
    uint z = id.z;
    uint batch = z / channels;
    uint channel = z % channels;
    
    if (x >= width || y >= height || batch >= batchSize) {
        return;
    }
    
    uint idx = (batch * channels * height * width) +
               (channel * height * width) +
               (y * width) + x;
    
    float lambda = mixingFactors[batch];
    output[idx] = lambda * inputA[idx] + (1.0 - lambda) * inputB[idx];
}