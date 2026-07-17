// Fake-SSS skin shader for the Anya demo (Built-in RP, forward).
// Why: Standard lighting makes skin read as painted plastic — real skin scatters light, so its
// terminator (light->shadow transition) is soft and blushes red. This shader approximates that with
// wrap ("half-Lambert") diffuse whose transition is tinted by a subsurface color, plus a tiling
// pore-detail normal so close-ups aren't rubber-smooth, plus a fresnel sheen for the oily film.
Shader "DeepUnity/AnyaSkin"
{
    Properties
    {
        _MainTex ("Albedo", 2D) = "white" {}
        _BumpMap ("Normal", 2D) = "bump" {}
        _BumpScale ("Normal Strength", Range(0,2)) = 1
        _DetailNormal ("Pore Detail Normal", 2D) = "bump" {}
        _DetailTiling ("Pore Tiling", Float) = 14
        _DetailScale ("Pore Strength", Range(0,2)) = 0.55
        _SpecGlossMap ("Specular (RGB) Gloss (A)", 2D) = "black" {}
        _SpecIntensity ("Spec Intensity", Range(0,2)) = 0.9
        _Shininess ("Shininess", Range(2,256)) = 48
        _SSSColor ("Subsurface Tint", Color) = (0.72, 0.22, 0.14, 1)
        _Wrap ("Diffuse Wrap", Range(0,1)) = 0.4
        _SheenColor ("Fresnel Sheen", Color) = (0.16, 0.15, 0.14, 1)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 300

        CGPROGRAM
        #pragma surface surf Skin fullforwardshadows
        #pragma target 3.0

        sampler2D _MainTex, _BumpMap, _DetailNormal, _SpecGlossMap;
        half _BumpScale, _DetailTiling, _DetailScale, _SpecIntensity, _Shininess, _Wrap;
        fixed4 _SSSColor, _SheenColor;

        struct Input { float2 uv_MainTex; float3 viewDir; };

        struct SurfaceOutputSkin
        {
            fixed3 Albedo;
            fixed3 Normal;
            fixed3 Emission;
            half3 SpecCol;
            half Gloss;
            half Alpha;
        };

        void surf (Input IN, inout SurfaceOutputSkin o)
        {
            fixed4 c = tex2D(_MainTex, IN.uv_MainTex);
            o.Albedo = c.rgb;

            half3 baseN = UnpackScaleNormal(tex2D(_BumpMap, IN.uv_MainTex), _BumpScale);
            half3 poreN = UnpackScaleNormal(tex2D(_DetailNormal, IN.uv_MainTex * _DetailTiling), _DetailScale);
            o.Normal = normalize(half3(baseN.xy + poreN.xy, baseN.z * poreN.z));   // whiteout-ish blend

            fixed4 sg = tex2D(_SpecGlossMap, IN.uv_MainTex);
            o.SpecCol = sg.rgb * _SpecIntensity;
            o.Gloss = sg.a;

            // fresnel sheen — the thin oily film catching rim light
            half fres = pow(1 - saturate(dot(normalize(IN.viewDir), o.Normal)), 3);
            o.Emission = _SheenColor.rgb * fres * c.rgb;
            o.Alpha = 1;
        }

        half4 LightingSkin (SurfaceOutputSkin s, half3 lightDir, half3 viewDir, half atten)
        {
            half NdotL = dot(s.Normal, lightDir);

            // wrap diffuse: light reaches a bit past the geometric terminator (scattering)
            half diff = saturate((NdotL + _Wrap) / (1 + _Wrap));
            // the terminator blushes toward the subsurface tint, fading out where fully lit
            half3 scatter = lerp(_SSSColor.rgb * 1.6, half3(1,1,1), smoothstep(0.0, 0.65, diff));

            // Blinn-Phong specular off the spec map (dual-scale-ish via gloss in alpha)
            half3 h = normalize(lightDir + viewDir);
            half nh = saturate(dot(s.Normal, h));
            half spec = pow(nh, _Shininess * max(s.Gloss, 0.05)) * step(0, NdotL);

            half3 col = s.Albedo * _LightColor0.rgb * diff * scatter * atten
                      + _LightColor0.rgb * s.SpecCol * spec * atten;
            return half4(col, 1);
        }
        ENDCG
    }
    FallBack "Standard"
}
