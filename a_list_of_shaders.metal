#include <metal_stdlib>
using namespace metal;

// Basic metallic surface shader
struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float2 texCoord [[attribute(2)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 worldPos;
    float3 normal;
    float2 texCoord;
};

struct Uniforms {
    float4x4 modelMatrix;
    float4x4 viewProjectionMatrix;
    float3 cameraPosition;
};

// Vertex shader
vertex VertexOut metallic_vertex(VertexIn in [[stage_in]],
                                constant Uniforms& uniforms [[buffer(0)]]) {
    VertexOut out;
    float4 worldPosition = uniforms.modelMatrix * float4(in.position, 1.0);
    out.position = uniforms.viewProjectionMatrix * worldPosition;
    out.worldPos = worldPosition.xyz;
    out.normal = normalize((uniforms.modelMatrix * float4(in.normal, 0.0)).xyz);
    out.texCoord = in.texCoord;
    return out;
}

// Fragment shader with PBR calculation
fragment float4 metallic_fragment(VertexOut in [[stage_in]],
                                constant Uniforms& uniforms [[buffer(0)]],
                                texture2d<float> albedoMap [[texture(0)]],
                                texture2d<float> metallicMap [[texture(1)]],
                                texture2d<float> roughnessMap [[texture(2)]]) {
    constexpr sampler textureSampler(filter::linear);
    
    float3 albedo = albedoMap.sample(textureSampler, in.texCoord).rgb;
    float metallic = metallicMap.sample(textureSampler, in.texCoord).r;
    float roughness = roughnessMap.sample(textureSampler, in.texCoord).r;
    
    float3 N = normalize(in.normal);
    float3 V = normalize(uniforms.cameraPosition - in.worldPos);
    
    // Calculate base reflectivity
    float3 F0 = mix(float3(0.04), albedo, metallic);
    
    // Simple directional light
    float3 L = normalize(float3(1, 1, 1));
    float3 H = normalize(V + L);
    
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);
    
    // Specular term
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
    float D = alpha2 / (M_PI_F * denom * denom);
    
    // Fresnel term
    float3 F = F0 + (1.0 - F0) * pow(1.0 - HdotV, 5.0);
    
    // Geometric term
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float G = NdotV / (NdotV * (1.0 - k) + k);
    
    float3 specular = (D * F * G) / (4.0 * NdotV * NdotL);
    float3 diffuse = albedo * (1.0 - metallic);
    
    float3 finalColor = (diffuse / M_PI_F + specular) * NdotL;
    
    return float4(finalColor, 1.0);
}

// Chrome-like reflective metal shader
fragment float4 chrome_fragment(VertexOut in [[stage_in]],
                              constant Uniforms& uniforms [[buffer(0)]],
                              texturecube<float> envMap [[texture(0)]]) {
    float3 N = normalize(in.normal);
    float3 V = normalize(uniforms.cameraPosition - in.worldPos);
    float3 R = reflect(-V, N);
    
    constexpr sampler envMapSampler(filter::linear, mip_filter::linear);
    float3 envColor = envMap.sample(envMapSampler, R).rgb;
    
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 5.0);
    float3 finalColor = mix(float3(0.95), envColor, fresnel);
    
    return float4(finalColor, 1.0);
}

// Anisotropic metal shader for brushed metal effect
fragment float4 anisotropic_fragment(VertexOut in [[stage_in]],
                                   constant Uniforms& uniforms [[buffer(0)]]) {
    float3 N = normalize(in.normal);
    float3 T = normalize(float3(1.0, 0.0, 0.0)); // Tangent direction
    float3 B = normalize(cross(N, T));
    
    float3 V = normalize(uniforms.cameraPosition - in.worldPos);
    float3 L = normalize(float3(1.0, 1.0, 1.0));
    float3 H = normalize(V + L);
    
    float NH = max(dot(N, H), 0.0);
    float NL = max(dot(N, L), 0.0);
    float NV = max(dot(N, V), 0.0);
    
    float TX = dot(T, H);
    float BX = dot(B, H);
    
    float roughnessX = 0.2; // Along tangent
    float roughnessY = 0.5; // Along bitangent
    
    float exponent = (TX * TX) / (roughnessX * roughnessX) + 
                    (BX * BX) / (roughnessY * roughnessY);
    
    float spec = exp(-exponent) * NL;
    return float4(float3(0.9) * (0.1 + 0.9 * spec), 1.0);
}

// Iridescent metal shader
fragment float4 iridescent_fragment(VertexOut in [[stage_in]],
                                  constant Uniforms& uniforms [[buffer(0)]]) {
    float3 N = normalize(in.normal);
    float3 V = normalize(uniforms.cameraPosition - in.worldPos);
    float NV = max(dot(N, V), 0.0);
    
    // Create rainbow effect based on view angle
    float3 baseColor = float3(0.8);
    float theta = acos(NV);
    
    float3 rainbow;
    rainbow.r = 0.5 + 0.5 * sin(theta * 6.0);
    rainbow.g = 0.5 + 0.5 * sin(theta * 6.0 + 2.094);
    rainbow.b = 0.5 + 0.5 * sin(theta * 6.0 + 4.189);
    
    float3 finalColor = mix(baseColor, rainbow, 0.5);
    return float4(finalColor, 1.0);
}

// Performance-optimized metal shader
fragment float4 fast_metal_fragment(VertexOut in [[stage_in]],
                                  constant Uniforms& uniforms [[buffer(0)]]) {
    float3 N = normalize(in.normal);
    float3 V = normalize(uniforms.cameraPosition - in.worldPos);
    float3 L = normalize(float3(1.0, 1.0, 1.0));
    float3 H = normalize(V + L);
    
    float NdotH = max(dot(N, H), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    
    float roughness = 0.3;
    float specPower = exp2(10.0 * (1.0 - roughness));
    float spec = pow(NdotH, specPower);
    
    float3 baseColor = float3(0.95, 0.93, 0.88);
    float3 finalColor = baseColor * (0.2 + 0.8 * spec * NdotL);
    
    return float4(finalColor, 1.0);
}

fragment float4 toon_fragment(VertexOut in [[stage_in]],
                            constant Uniforms& uniforms [[buffer(0)]],
                            texture2d<float> diffuseMap [[texture(0)]]) {
    constexpr sampler textureSampler(filter::linear);
    float3 baseColor = diffuseMap.sample(textureSampler, in.texCoord).rgb;
    
    float3 N = normalize(in.normal);
    float3 L = normalize(float3(1.0, 1.0, 1.0));
    float NdotL = dot(N, L);
    
    // Create cel-shading steps
    float cel = NdotL > 0.75 ? 1.0 :
                NdotL > 0.45 ? 0.75 :
                NdotL > 0.25 ? 0.5 :
                0.25;
    
    // Add rim lighting
    float3 V = normalize(uniforms.cameraPosition - in.worldPos);
    float rim = 1.0 - max(dot(V, N), 0.0);
    rim = pow(rim, 4.0);
    
    float3 finalColor = baseColor * cel + float3(1.0) * rim;
    return float4(finalColor, 1.0);
}

// Dissolve effect shader for transitions
fragment float4 dissolve_fragment(VertexOut in [[stage_in]],
                                constant float& dissolveThreshold [[buffer(1)]],
                                texture2d<float> diffuseMap [[texture(0)]],
                                texture2d<float> noiseMap [[texture(1)]]) {
    constexpr sampler textureSampler(filter::linear);
    
    float4 baseColor = diffuseMap.sample(textureSampler, in.texCoord);
    float noise = noiseMap.sample(textureSampler, in.texCoord).r;
    
    // Create dissolve edge effect
    float edgeWidth = 0.1;
    float dissolveEdge = smoothstep(dissolveThreshold, dissolveThreshold + edgeWidth, noise);
    
    // Add glow color at dissolution edge
    float3 edgeColor = float3(1.0, 0.5, 0.0); // Orange glow
    float3 finalColor = mix(edgeColor, baseColor.rgb, dissolveEdge);
    
    // Clip fragments based on noise and threshold
    if (noise < dissolveThreshold) {
        discard_fragment();
    }
    
    return float4(finalColor, baseColor.a);
}

// Water surface shader with waves and reflection
fragment float4 water_fragment(VertexOut in [[stage_in]],
                             constant Uniforms& uniforms [[buffer(0)]],
                             texture2d<float> normalMap [[texture(0)]],
                             texturecube<float> environmentMap [[texture(1)]],
                             constant float& time [[buffer(1)]]) {
    constexpr sampler textureSampler(filter::linear, address::repeat);
    
    // Animate UVs for flowing water effect
    float2 uv1 = in.texCoord + float2(time * 0.1, time * 0.05);
    float2 uv2 = in.texCoord * 2.0 - float2(time * 0.15, time * 0.07);
    
    // Sample normal maps with different scales and animations
    float3 normal1 = normalMap.sample(textureSampler, uv1).rgb * 2.0 - 1.0;
    float3 normal2 = normalMap.sample(textureSampler, uv2).rgb * 2.0 - 1.0;
    
    // Blend normals
    float3 finalNormal = normalize(normal1 + normal2);
    finalNormal = normalize(float3(finalNormal.xy * 0.5, finalNormal.z));
    
    // Calculate reflection
    float3 V = normalize(uniforms.cameraPosition - in.worldPos);
    float3 R = reflect(-V, finalNormal);
    
    // Sample environment map for reflection
    float3 reflection = environmentMap.sample(textureSampler, R).rgb;
    
    // Fresnel effect
    float fresnel = pow(1.0 - max(dot(finalNormal, V), 0.0), 4.0);
    
    // Final color combining reflection and water color
    float3 waterColor = float3(0.0, 0.3, 0.5);
    float3 finalColor = mix(waterColor, reflection, fresnel);
    
    return float4(finalColor, 0.9);
}

// Hologram effect shader
fragment float4 hologram_fragment(VertexOut in [[stage_in]],
                                constant float& time [[buffer(1)]]) {
    // Base hologram color
    float3 hologramColor = float3(0.0, 1.0, 0.9); // Cyan tint
    
    // Scan line effect
    float scanLine = fract(in.worldPos.y * 20.0 + time * 2.0);
    scanLine = smoothstep(0.0, 0.1, scanLine) * smoothstep(1.0, 0.9, scanLine);
    
    // Flickering effect
    float flicker = sin(time * 8.0) * 0.1 + 0.9;
    
    // Edge highlight
    float3 N = normalize(in.normal);
    float3 V = normalize(float3(0.0, 0.0, 1.0));
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 2.0);
    
    // Glitch effect
    float glitch = step(0.98, fract(time * 3.0));
    float glitchOffset = (fract(in.worldPos.y * 5.0) - 0.5) * glitch;
    
    float3 finalColor = hologramColor * (scanLine * 0.5 + 0.5) * flicker;
    finalColor += hologramColor * fresnel * 0.5;
    finalColor += glitchOffset;
    
    // Alpha for transparency
    float alpha = 0.7 * flicker;
    
    return float4(finalColor, alpha);
}

// Outline shader (requires separate vertex shader)
vertex VertexOut outline_vertex(VertexIn in [[stage_in]],
                              constant Uniforms& uniforms [[buffer(0)]],
                              constant float& outlineWidth [[buffer(1)]]) {
    VertexOut out;
    
    // Expand vertices along normals
    float3 expandedPos = in.position + in.normal * outlineWidth;
    float4 worldPosition = uniforms.modelMatrix * float4(expandedPos, 1.0);
    out.position = uniforms.viewProjectionMatrix * worldPosition;
    
    out.worldPos = worldPosition.xyz;
    out.normal = in.normal;
    out.texCoord = in.texCoord;
    
    return out;
}

fragment float4 outline_fragment() {
    return float4(0.0, 0.0, 0.0, 1.0); // Solid black outline
}

// Force field effect shader
fragment float4 force_field_fragment(VertexOut in [[stage_in]],
                                   constant Uniforms& uniforms [[buffer(0)]],
                                   constant float& time [[buffer(1)]]) {
    float3 N = normalize(in.normal);
    float3 V = normalize(uniforms.cameraPosition - in.worldPos);
    
    // Hexagonal pattern
    float2 hex = in.texCoord * float2(8.0, 16.0);
    hex.x *= 0.866025404; // sqrt(3)/2
    hex.y += hex.x * 0.5;
    
    float2 hexInt = floor(hex);
    float2 hexFract = fract(hex);
    
    // Animate pattern
    float hexPattern = sin(hexInt.x + hexInt.y + time * 2.0) * 0.5 + 0.5;
    
    // Fresnel effect
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    
    // Pulse wave
    float pulse = sin(time * 3.0 + in.worldPos.y * 4.0) * 0.5 + 0.5;
    
    float3 baseColor = float3(0.2, 0.6, 1.0);
    float3 finalColor = baseColor * (fresnel + hexPattern * 0.5 + pulse * 0.2);
    
    return float4(finalColor, 0.7);
}

// Shield impact effect shader
fragment float4 shield_impact_fragment(VertexOut in [[stage_in]],
                                     constant float3& impactPoint [[buffer(1)]],
                                     constant float& impactTime [[buffer(2)]]) {
    float3 N = normalize(in.normal);
    
    // Calculate distance from impact point
    float dist = length(in.worldPos - impactPoint);
    
    // Create expanding ring effect
    float ringRadius = impactTime * 2.0;
    float ringWidth = 0.2;
    float ring = smoothstep(0.0, ringWidth, abs(dist - ringRadius));
    
    // Impact pattern
    float pattern = sin(dist * 20.0 - impactTime * 10.0) * 0.5 + 0.5;
    
    float3 shieldColor = float3(0.3, 0.7, 1.0);
    float3 impactColor = float3(1.0, 0.3, 0.1);
    
    float3 finalColor = mix(impactColor, shieldColor, ring);
    finalColor += pattern * 0.2;
    
    return float4(finalColor, 0.8);
}

// Pixelation effect shader
fragment float4 pixelate_fragment(VertexOut in [[stage_in]],
                                texture2d<float> sourceTexture [[texture(0)]],
                                constant float& pixelSize [[buffer(1)]]) {
    constexpr sampler textureSampler(filter::nearest);
    
    // Calculate pixelated UV coordinates
    float2 pixelatedUV = floor(in.texCoord * pixelSize) / pixelSize;
    
    return sourceTexture.sample(textureSampler, pixelatedUV);
}

// Heat distortion effect shader
fragment float4 heat_distortion_fragment(VertexOut in [[stage_in]],
                                       texture2d<float> sourceTexture [[texture(0)]],
                                       texture2d<float> noiseTexture [[texture(1)]],
                                       constant float& time [[buffer(1)]]) {
    constexpr sampler textureSampler(filter::linear);
    
    // Animate noise texture coordinates
    float2 noiseUV = in.texCoord + float2(time * 0.1, time * 0.2);
    
    // Sample noise for distortion
    float2 noise = noiseTexture.sample(textureSampler, noiseUV).rg * 2.0 - 1.0;
    
    // Apply distortion to texture coordinates
    float2 distortedUV = in.texCoord + noise * 0.02;
    
    // Sample source texture with distorted coordinates
    float4 color = sourceTexture.sample(textureSampler, distortedUV);
    
    // Add heat haze color
    float3 heatColor = float3(1.0, 0.5, 0.2);
    return float4(color.rgb + heatColor * length(noise) * 0.1, color.a);
}

// Terrain blend shader for smooth transitions between textures
fragment float4 terrain_blend_fragment(VertexOut in [[stage_in]],
                                     texture2d<float> texture1 [[texture(0)]],
                                     texture2d<float> texture2 [[texture(1)]],
                                     texture2d<float> blendMap [[texture(2)]]) {
    constexpr sampler textureSampler(filter::linear, address::repeat);
    
    // Sample textures at different scales
    float2 terrainUV = in.texCoord * 8.0;
    float4 color1 = texture1.sample(textureSampler, terrainUV);
    float4 color2 = texture2.sample(textureSampler, terrainUV);
    
    // Sample blend map
    float blend = blendMap.sample(textureSampler, in.texCoord).r;
    
    // Add height-based blending
    float heightBlend = smoothstep(0.2, 0.8, in.worldPos.y * 0.1);
    blend = mix(blend, heightBlend, 0.5);
    
    return mix(color1, color2, blend);
}

// X-Ray vision effect shader
fragment float4 xray_fragment(VertexOut in [[stage_in]],
                            constant Uniforms& uniforms [[buffer(0)]]) {
    float3 N = normalize(in.normal);
    float3 V = normalize(uniforms.cameraPosition - in.worldPos);
    
    // Edge detection based on normal and view angle
    float edge = 1.0 - abs(dot(N, V));
    edge = pow(edge, 2.0);
    
    // Scan line effect
    float scanLine = fract(in.worldPos.y * 30.0) * 0.5 + 0.5;
    
    // Depth-based intensity
    float depth = length(uniforms.cameraPosition - in.worldPos);
    float depthFade = exp(-depth * 0.1);
    
    float3 xrayColor = float3(0.0, 1.0, 0.8) * edge * scanLine * depthFade;
    return float4(xrayColor, 0.7);
}

// Glass shader with refraction and chromatic aberration
fragment float4 glass_fragment(VertexOut in [[stage_in]],
                             constant Uniforms& uniforms [[buffer(0)]],
                             texturecube<float> envMap [[texture(0)]]) {
    float3 N = normalize(in.normal);
    float3 V = normalize(uniforms.cameraPosition - in.worldPos);
    
    // Index of refraction for different color channels
    float iorR = 1.1;
    float iorG = 1.12;
    float iorB = 1.15;
    
    // Calculate refraction for each color channel
    float3 refractionR = refract(-V, N, 1.0/iorR);
    float3 refractionG = refract(-V, N, 1.0/iorG);
    float3 refractionB = refract(-V, N, 1.0/iorB);
    
    constexpr sampler envMapSampler(filter::linear);
    float r = envMap.sample(envMapSampler, refractionR).r;
    float g = envMap.sample(envMapSampler, refractionG).g;
    float b = envMap.sample(envMapSampler, refractionB).b;
    
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 4.0);
    float3 reflection = envMap.sample(envMapSampler, reflect(-V, N)).rgb;
    
    return float4(mix(float3(r, g, b), reflection, fresnel), 0.9);
}

// Paint drip effect shader
fragment float4 paint_drip_fragment(VertexOut in [[stage_in]],
                                  constant float& time [[buffer(1)]],
                                  texture2d<float> noiseMap [[texture(0)]]) {
    constexpr sampler noiseSampler(filter::linear);
    
    // Base paint color
    float3 paintColor = float3(0.8, 0.2, 0.3);
    
    // Create drip pattern
    float2 dripUV = in.texCoord;
    dripUV.y -= time * 0.1;
    
    float noise = noiseMap.sample(noiseSampler, dripUV).r;
    float drip = smoothstep(0.4, 0.6, noise);
    
    // Add variation to drip speed based on position
    float dripSpeed = sin(in.texCoord.x * 10.0) * 0.5 + 0.5;
    float dripPattern = smoothstep(0.0, 0.1, fract(dripUV.y + dripSpeed));
    
    float3 finalColor = paintColor * (drip * dripPattern + 0.5);
    return float4(finalColor, 1.0);
}

// Lightning effect shader
fragment float4 lightning_fragment(VertexOut in [[stage_in]],
                                 constant float& time [[buffer(1)]]) {
    // Generate lightning pattern
    float2 uv = in.texCoord * 2.0 - 1.0;
    float lightning = 0.0;
    
    // Create branching effect
    for(int i = 0; i < 5; i++) {
        float t = time * (1.0 + float(i) * 0.5);
        float2 offset = float2(sin(t * 2.0), cos(t * 3.0)) * 0.2;
        float d = length(uv + offset);
        lightning += 0.1 / (d * d);
    }
    
    // Add flickering
    float flicker = sin(time * 30.0) * 0.5 + 0.5;
    
    float3 lightningColor = float3(0.7, 0.8, 1.0);
    return float4(lightningColor * lightning * flicker, 1.0);
}

// Portal effect shader
fragment float4 portal_fragment(VertexOut in [[stage_in]],
                              constant float& time [[buffer(1)]]) {
    // Create spiral UV coordinates
    float2 uv = in.texCoord * 2.0 - 1.0;
    float angle = atan2(uv.y, uv.x);
    float radius = length(uv);
    
    // Spiral pattern
    float spiral = sin(angle * 5.0 + radius * 10.0 - time * 3.0);
    
    // Add swirling effect
    float swirl = sin(radius * 10.0 - time * 2.0);
    
    // Edge glow
    float edge = 1.0 - smoothstep(0.0, 1.0, abs(radius - 0.5));
    
    float3 portalColor = float3(0.5, 0.0, 1.0);
    float3 glowColor = float3(1.0, 0.5, 0.0);
    
    float3 finalColor = mix(portalColor, glowColor, spiral * swirl) * edge;
    return float4(finalColor, edge);
}

// Glitch art shader
fragment float4 glitch_art_fragment(VertexOut in [[stage_in]],
                                  texture2d<float> sourceTexture [[texture(0)]],
                                  constant float& time [[buffer(1)]]) {
    constexpr sampler textureSampler(filter::linear);
    
    // Create glitch offset
    float2 uv = in.texCoord;
    float glitchStrength = sin(time * 10.0) * 0.5 + 0.5;
    
    // RGB shift
    float2 redOffset = uv + float2(0.01, 0.0) * glitchStrength;
    float2 blueOffset = uv - float2(0.01, 0.0) * glitchStrength;
    
    // Random blocks
    float2 blockOffset = floor(uv * 20.0) / 20.0;
    float random = fract(sin(dot(blockOffset, float2(12.9898, 78.233))) * 43758.5453);
    
    // Sample with offsets
    float r = sourceTexture.sample(textureSampler, redOffset).r;
    float g = sourceTexture.sample(textureSampler, uv).g;
    float b = sourceTexture.sample(textureSampler, blueOffset).b;
    
    // Add noise
    float noise = random * glitchStrength * 0.2;
    
    return float4(r + noise, g + noise, b + noise, 1.0);
}

// Ink dispersion shader
fragment float4 ink_dispersion_fragment(VertexOut in [[stage_in]],
                                      constant float& time [[buffer(1)]],
                                      texture2d<float> noiseTexture [[texture(0)]]) {
    constexpr sampler noiseSampler(filter::linear);
    
    // Animate noise coordinates
    float2 noiseUV = in.texCoord + float2(time * 0.1);
    float noise = noiseTexture.sample(noiseSampler, noiseUV).r;
    
    // Create flowing patterns
    float flow = sin(in.texCoord.y * 10.0 + time + noise * 5.0);
    
    // Ink color gradient
    float3 inkColor1 = float3(0.0, 0.0, 0.1);
    float3 inkColor2 = float3(0.2, 0.0, 0.3);
    
    float3 finalColor = mix(inkColor1, inkColor2, flow * noise);
    float alpha = smoothstep(0.2, 0.8, noise);
    
    return float4(finalColor, alpha);
}

// Neon glow shader
fragment float4 neon_glow_fragment(VertexOut in [[stage_in]],
                                 constant Uniforms& uniforms [[buffer(0)]]) {
    float3 N = normalize(in.normal);
    float3 V = normalize(uniforms.cameraPosition - in.worldPos);
    
    // Edge detection
    float edge = 1.0 - abs(dot(N, V));
    edge = pow(edge, 3.0);
    
    // Pulse effect
    float pulse = sin(uniforms.time * 2.0) * 0.5 + 0.5;
    
    // Core and glow colors
    float3 coreColor = float3(1.0, 0.2, 0.8);
    float3 glowColor = float3(0.5, 0.0, 1.0);
    
    float3 finalColor = mix(coreColor, glowColor, edge) * (1.0 + pulse * 0.5);
    return float4(finalColor, 1.0);
}